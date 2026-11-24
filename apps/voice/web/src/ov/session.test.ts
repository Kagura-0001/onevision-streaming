import { describe, expect, it } from "vitest";

import { ANSWER_QUIET_MS, OvSession, QUESTION_WAIT_MS, joinText } from "./session";

function setup() {
  let t = 0;
  const s = new OvSession(() => t);
  const at = (ms: number) => {
    t = ms;
  };
  const ov = (data: Record<string, unknown>) => s.onServerMessage({ ov: "event", ...data });
  const question = (input_id: string, text: string, segments = 1) =>
    s.onServerMessage({ ov: "question", input_id, text, segments, timestamp_sec: 1 });
  return { s, at, ov, question, snap: () => s.getSnapshot() };
}

describe("OvSession", () => {
  it("walks a spoken question through to OV's spoken answer", () => {
    const { s, at, ov, question, snap } = setup();
    s.userStarted();
    expect(snap().turns[0].status).toBe("检测到你在说话");
    s.userTranscript("这锅水开了吗？");
    at(1000);
    s.userStopped();
    question("in1", "这锅水开了吗？");
    expect(snap().turns[0]).toMatchObject({ step: 2, status: "已发送给 OV", inputId: "in1" });

    ov({ type: "response.created", input_id: "in1", response_id: "r1", control: "response" });
    ov({ type: "input.done", input_id: "in1", response_id: "r1", control: "response", latency_ms: 55 });
    expect(snap().turns[0]).toMatchObject({ step: 3, status: "OV 决定回答" });
    expect(snap().turns[0].note).toContain("OV 用时 55 ms");

    ov({ type: "response.output_text.done", response_id: "r1", control: "response", text: "At 3 seconds: 还没有开。快了。" });
    expect(snap().responses[0]).toMatchObject({ state: "waiting", turnId: snap().turns[0].id });

    s.botStarted();
    s.botSaid("还没有开。", true);
    expect(snap().responses[0].state).toBe("speaking");
    expect(snap().turns[0].step).toBe(4);
    s.botStopped();
    s.botStarted();
    s.botSaid("快了。", true);
    s.botStopped();
    expect(snap().responses[0].state).toBe("spoken");
    expect(snap().turns[0]).toMatchObject({ status: "OV 说完了", final: true });
    expect(snap().stats).toMatchObject({ inputs: 1, answers: 1, latencies: [55], strip: ["response"] });
  });

  it("joins the transcripts of one turn into one question", () => {
    const { s, question, snap } = setup();
    s.userStarted();
    s.userTranscript("你好。");
    s.userTranscript("你看，这锅水开了吗？");
    s.userStopped();
    expect(snap().turns).toHaveLength(1);
    expect(snap().turns[0].text).toBe("你好。你看，这锅水开了吗？");
    question("in1", "你好。你看，这锅水开了吗？", 2);
    expect(snap().turns[0].note).toContain("2 段合并");
  });

  it("marks a turn OV stays silent on as finished", () => {
    const { s, ov, question, snap } = setup();
    s.userStarted();
    s.userTranscript("嗯");
    s.userStopped();
    question("in1", "嗯");
    ov({ type: "input.done", input_id: "in1", control: "silence", latency_ms: 50 });
    expect(snap().turns[0]).toMatchObject({ tone: "silent", status: "OV 选择不回答", final: true });
  });

  it("keeps OV's own captions, tied to no turn", () => {
    const { ov, snap } = setup();
    ov({ type: "response.created", input_id: "f20", response_id: "c1", control: "caption" });
    ov({ type: "response.output_text.done", response_id: "c1", control: "caption", text: "At 0 - 20 seconds: 厨房里…" });
    expect(snap().turns).toHaveLength(0);
    expect(snap().responses[0]).toMatchObject({ control: "caption", state: "caption", turnId: null });
  });

  it("lets a newer answer cut the one playing, and the user cut the newest", () => {
    const { s, ov, snap } = setup();
    ov({ type: "response.created", input_id: "a", response_id: "r1", control: "response" });
    ov({ type: "response.output_text.done", response_id: "r1", control: "response", text: "第一句。" });
    s.botStarted();
    s.botSaid("第一句。", false);
    ov({ type: "response.created", input_id: "b", response_id: "r2", control: "response" });
    ov({ type: "response.output_text.done", response_id: "r2", control: "response", text: "第二句。" });
    s.botSaid("第二句。", false);
    s.userStarted();
    const byId = Object.fromEntries(snap().responses.map((r) => [r.id, r.state]));
    expect(byId).toEqual({ r1: "superseded", r2: "interrupted" });
  });

  it("applies an input.done that arrives before its question", () => {
    const { s, ov, question, snap } = setup();
    s.userStarted();
    s.userTranscript("看");
    s.userStopped();
    ov({ type: "input.done", input_id: "in1", control: "silence", latency_ms: 40 });
    question("in1", "看");
    expect(snap().turns[0]).toMatchObject({ step: 3, status: "OV 选择不回答" });
  });

  it("says so when a turn ends with no transcript", () => {
    const { s, at, snap } = setup();
    s.userStarted();
    at(500);
    s.userStopped();
    at(500 + QUESTION_WAIT_MS + 1);
    s.tick();
    expect(snap().turns[0]).toMatchObject({ tone: "warn", status: "没有识别出文字", final: true });
  });

  it("ends an answer once OV has been quiet for a while", () => {
    const { s, at, ov, snap } = setup();
    ov({ type: "response.created", input_id: "a", response_id: "r1", control: "response" });
    ov({ type: "response.output_text.done", response_id: "r1", control: "response", text: "一句。另一句。" });
    s.botStarted();
    s.botSaid("一句。", false);
    at(100);
    s.botStopped();
    at(100 + ANSWER_QUIET_MS + 1);
    s.tick();
    expect(snap().responses[0].state).toBe("spoken");
  });

  it("turns a typed question into the question the bot sends", () => {
    const { s, question, snap } = setup();
    s.typed("你看到了什么？");
    question("in1", "你看到了什么？");
    expect(snap().turns).toHaveLength(1);
    expect(snap().turns[0]).toMatchObject({ inputId: "in1", step: 2 });
  });
});

describe("joinText", () => {
  it("puts a space only between words", () => {
    expect(joinText("这锅水", "开了吗？")).toBe("这锅水开了吗？");
    expect(joinText("is the pot", "boiling?")).toBe("is the pot boiling?");
    expect(joinText("", "hello")).toBe("hello");
  });
});
