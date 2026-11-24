// What the page knows about one session, built only from the messages the bot
// sends: RTVI events (speech, transcripts, OV speaking) and OV's own events,
// which OVProcessor forwards as RTVI server messages ({ ov: "question" | "event" }).
//
// A user turn moves through STEPS. It is matched to its question by order (the
// bot sends every finished turn not yet sent), to OV's decision by input_id,
// and to the answer playing by the sentence text the bot reports. tick() says
// when a turn stalls. OV's answers and captions are kept per response, so the
// page shows one conversation, not a turn list and an output log side by side.

export const STEPS = ["说话", "识别", "发给 OV", "OV 决定", "OV 说"] as const;

export const QUESTION_WAIT_MS = 6000; // turn over -> question sent; normally within 3 s
export const DECIDE_WAIT_MS = 8000; // question sent -> OV's decision; normally under 1 s
export const LONG_TURN_MS = 20000; // the server still thinks the user is talking
export const ANSWER_QUIET_MS = 2000; // OV silent this long after a sentence: its answer is over
export const ANSWER_WAIT_MS = 8000; // a written answer not played this long, with nothing in its way
const KEEP_TURNS = 30;
const KEEP_STRIP = 60;

export type Tone = "you" | "ov" | "caption" | "silent" | "warn";

export interface Turn {
  id: number;
  t0: number;
  ended: number | null;
  sentAt: number | null;
  finishedAt: number | null;
  text: string;
  segments: number;
  inputId: string | null;
  responseId: string | null;
  step: number;
  tone: Tone;
  status: string;
  note: string;
  final: boolean;
  warned: boolean;
}

export type ResponseState =
  | "writing" // OV is generating it
  | "waiting" // written, waiting to be played
  | "speaking"
  | "spoken"
  | "interrupted" // the user spoke while it played
  | "dropped" // never played: the user spoke again first
  | "superseded" // cut or never played: a newer answer came
  | "cancelled"
  | "failed"
  | "caption" // captions are written back, never spoken
  | "empty";

export interface OvResponse {
  id: string;
  control: string; // "response" or "caption"
  createdAt: number;
  timestampSec: number | null;
  inputId: string | null;
  turnId: number | null; // the user turn it answers; null when OV spoke up on its own
  text: string;
  textNorm: string;
  state: ResponseState;
  reason: string;
  readyAt: number | null;
  confirmed: boolean; // a sentence of it has been heard back from the bot
  lastSentence: boolean;
}

export interface Note {
  id: string;
  at: number;
  kind: "compacted" | "error";
  text: string;
}

export interface Stats {
  inputs: number;
  answers: number;
  latencies: number[];
  strip: string[]; // OV's control on each input, oldest first
  lastInputDoneAt: number | null;
}

export interface Snapshot {
  turns: readonly Turn[];
  responses: readonly OvResponse[];
  notes: readonly Note[];
  stats: Stats;
  serverHearing: boolean;
  botTalking: boolean;
  speakingResponseId: string | null;
  botReady: boolean;
}

const DECISIONS: Record<string, { tone: Tone; status: string; note: string; final?: boolean }> = {
  response: { tone: "ov", status: "OV 决定回答", note: "正在生成回答…" },
  caption: { tone: "caption", status: "OV 决定解说", note: "解说不念出来，写在下面" },
  silence: {
    tone: "silent",
    status: "OV 选择不回答",
    note: "OV 在这一帧决定不出声；之后的画面上它仍可能主动开口",
    final: true,
  },
};

const norm = (s: string) => (s || "").replace(/\s+/g, "");
const WIDE = /[ᄀ-ᅟ⺀-꓏가-힣豈-﫿︰-﹏＀-｠￠-￦]/;
// As the bot joins the transcripts of a turn: a space only between words.
export function joinText(before: string, after: string): string {
  return before && after && !WIDE.test(before.slice(-1)) && !WIDE.test(after[0])
    ? `${before} ${after}`
    : before + after;
}
const secs = (ms: number) => `${(Math.max(0, ms) / 1000).toFixed(1)} s`;

type Data = Record<string, any>;

export class OvSession {
  private turns: Turn[] = [];
  private nextTurnId = 1;
  private turnByInput = new Map<string, Turn>();
  private responses = new Map<string, OvResponse>();
  private earlyInputs = new Map<string, Data>(); // input.done that came before its question
  private toSpeak: OvResponse[] = []; // written answers waiting to be played, in order
  private speaking: OvResponse | null = null;
  private notes: Note[] = [];
  private stats: Stats = { inputs: 0, answers: 0, latencies: [], strip: [], lastInputDoneAt: null };
  serverHearing = false; // between user-started- and user-stopped-speaking
  botTalking = false;
  botQuietAt = -Infinity;
  botReady = false;
  private listeners = new Set<() => void>();
  private snapshot: Snapshot;

  constructor(private now: () => number = () => performance.now()) {
    this.snapshot = this.build();
  }

  // ---- React store plumbing ----------------------------------------------------
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };
  getSnapshot = () => this.snapshot;
  private changed() {
    this.snapshot = this.build();
    for (const listener of this.listeners) listener();
  }
  private build(): Snapshot {
    return {
      turns: this.turns.map((t) => ({ ...t })),
      responses: [...this.responses.values()].map((r) => ({ ...r })),
      notes: [...this.notes],
      stats: { ...this.stats, latencies: [...this.stats.latencies], strip: [...this.stats.strip] },
      serverHearing: this.serverHearing,
      botTalking: this.botTalking,
      speakingResponseId: this.speaking?.id ?? null,
      botReady: this.botReady,
    };
  }

  // ---- Messages from the bot ----------------------------------------------------
  /** An RTVI server message; OVProcessor's carry `ov`. */
  onServerMessage(data: Data) {
    if (!data || typeof data !== "object") return;
    this.botReady = true;
    if (data.ov === "question") this.questionSent(data);
    else if (data.ov === "event") this.onOvEvent(data);
    else return;
    this.changed();
  }

  private onOvEvent(d: Data) {
    switch (d.type) {
      case "input.done":
        // A question may go in alone, without a frame; the rest carry one.
        if (!this.turnByInput.has(d.input_id)) this.stats.lastInputDoneAt = this.now();
        this.stats.inputs += 1;
        this.stats.strip.push(d.control || "silence");
        if (this.stats.strip.length > KEEP_STRIP) this.stats.strip.shift();
        if (typeof d.latency_ms === "number") {
          this.stats.latencies.push(d.latency_ms);
          if (this.stats.latencies.length > 100) this.stats.latencies.shift();
        }
        this.inputDone(d);
        break;
      case "response.created":
        this.responseCreated(d);
        break;
      case "response.output_text.done":
        this.answerWritten(d);
        break;
      case "response.done":
        this.responseDone(d);
        break;
      case "context.compacted":
        this.note("compacted", `上下文压缩：保留 ${d.retained_turns} 轮、${d.retained_visual_frames} 帧`);
        break;
      case "error":
        this.note("error", d.error || "OV 会话出错");
        break;
    }
  }

  userStarted() {
    this.serverHearing = true;
    // The bot stops the answer being played and drops the ones waiting.
    if (this.speaking) this.endSpeaking("interrupted");
    for (const r of this.toSpeak.splice(0)) this.settle(r, "dropped");
    if (!this.newest((t) => t.ended == null && !t.inputId)) this.newTurn();
    this.changed();
  }

  userStopped() {
    this.serverHearing = false;
    const turn = this.newest((t) => t.ended == null && !t.inputId);
    if (turn) {
      turn.ended = this.now();
      turn.warned = false;
      if (turn.text) turn.note = "这句说完了，随下一帧发给 OV";
      else turn.status = "你说完了，正在识别…";
    }
    this.changed();
  }

  userTranscript(text: string, final = true) {
    text = (text || "").trim();
    if (!text || !final) return;
    // A transcript belongs to the newest turn not yet sent. One that comes after
    // its turn's question goes in with the user's next turn, as the bot sends it.
    const turn =
      this.newest((t) => !t.inputId) || this.newTurn({ ended: this.serverHearing ? null : this.now() });
    this.update(turn, {
      text: joinText(turn.text, text),
      step: Math.max(turn.step, 1),
      tone: "you",
      status: "识别到",
      note: turn.ended != null ? "这句说完了，随下一帧发给 OV" : "",
      final: false,
      warned: false,
    });
    this.changed();
  }

  /** A question typed on the page, before the bot reports it sent. */
  typed(text: string) {
    const turn = this.newTurn({ ended: this.now(), text, step: 1, status: "打字发送", note: "随下一帧发给 OV" });
    turn.warned = false;
    this.changed();
  }

  private questionSent(d: Data) {
    // The bot sends every turn that has ended since its last question, joined.
    const waiting = this.turns.filter((t) => !t.inputId && t.ended != null);
    const heard = waiting.filter((t) => t.text);
    const joined = heard.length ? heard : waiting.slice(-1);
    const turn = joined[0] || this.newTurn();
    joined.slice(1).forEach((t) => this.dropTurn(t));
    turn.inputId = d.input_id;
    this.turnByInput.set(d.input_id, turn);
    turn.sentAt = this.now();
    turn.segments = d.segments || 1;
    const note = [
      turn.ended != null ? `说完后 ${secs(turn.sentAt - turn.ended)} 发出` : "",
      d.segments > 1 ? `${d.segments} 段合并` : "",
    ]
      .filter(Boolean)
      .join(" · ");
    this.update(turn, {
      text: d.text || turn.text,
      step: 2,
      tone: "you",
      status: "已发送给 OV",
      note,
      final: false,
      warned: false,
    });
    const early = this.earlyInputs.get(d.input_id);
    if (early) {
      this.earlyInputs.delete(d.input_id);
      this.inputDone(early);
    }
  }

  // OV reports response.created and then input.done for the input a question
  // went in with; either one carries the decision.
  private inputDone(d: Data) {
    const turn = this.turnByInput.get(d.input_id);
    if (!turn) {
      // The question may not have reached the page yet; keep the last few inputs.
      this.earlyInputs.set(d.input_id, d);
      while (this.earlyInputs.size > 30) this.earlyInputs.delete(this.earlyInputs.keys().next().value!);
      return;
    }
    this.link(turn, d.response_id);
    this.decide(turn, d.control, d.latency_ms ?? null);
  }

  private link(turn: Turn, responseId: string | undefined) {
    if (!responseId || turn.responseId) return;
    turn.responseId = responseId;
    const r = this.responses.get(responseId);
    if (r) r.turnId = turn.id;
  }

  private decide(turn: Turn, control: string, latencyMs: number | null) {
    const took = latencyMs != null ? `OV 用时 ${latencyMs} ms` : "";
    if (turn.step >= 3) {
      // Decided already, from response.created; add the time input.done brings.
      if (took && turn.step === 3 && !turn.final && !turn.note.startsWith("OV 用时")) {
        turn.note = [took, turn.note].filter(Boolean).join(" · ");
      }
      return;
    }
    const how = DECISIONS[control] || { tone: "ov" as Tone, status: `OV 决定：${control}`, note: "" };
    this.update(turn, {
      step: 3,
      warned: false,
      final: !!how.final,
      tone: how.tone,
      status: how.status,
      note: [took, how.note].filter(Boolean).join(" · "),
    });
  }

  private responseCreated(d: Data) {
    const turn = d.input_id ? this.turnByInput.get(d.input_id) : undefined;
    this.responses.set(d.response_id, {
      id: d.response_id,
      control: d.control,
      createdAt: this.now(),
      timestampSec: d.timestamp_sec ?? null,
      inputId: d.input_id ?? null,
      turnId: turn ? turn.id : null,
      text: "",
      textNorm: "",
      state: "writing",
      reason: "",
      readyAt: null,
      confirmed: false,
      lastSentence: false,
    });
    this.trimResponses();
    if (turn) {
      this.link(turn, d.response_id);
      this.decide(turn, d.control, null);
    }
  }

  private answerWritten(d: Data) {
    const r = this.responses.get(d.response_id);
    if (!r) return;
    const text = (d.text || "").trim();
    r.text = text;
    r.textNorm = norm(text);
    const turn = this.turnOf(r);
    if (d.control === "caption") {
      r.state = text ? "caption" : "empty";
      if (turn && !turn.final) this.update(turn, { status: "OV 解说了画面", note: "", final: true });
    } else if (!text) {
      r.state = "empty";
      if (turn && !turn.final) this.update(turn, { tone: "silent", status: "OV 的回答是空的", note: "", final: true });
    } else {
      this.stats.answers += 1;
      // The bot cuts the answer playing for a newer one and drops the one waiting.
      if (this.speaking) this.endSpeaking("superseded");
      for (const x of this.toSpeak.splice(0)) this.settle(x, "superseded");
      r.state = "waiting";
      r.readyAt = this.now();
      this.toSpeak.push(r);
      if (turn && !turn.final) this.update(turn, { status: "回答写好了，等待播放", note: "" });
    }
  }

  private responseDone(d: Data) {
    const r = this.responses.get(d.response_id);
    if (!r) return;
    if (d.status === "completed") {
      if (d.reason === "length") r.reason = "被截断，没有写回记忆";
      return;
    }
    this.toSpeak = this.toSpeak.filter((x) => x !== r);
    if (this.speaking === r) this.speaking = null;
    r.reason = d.reason || d.error || "";
    this.settle(r, d.status === "cancelled" ? "cancelled" : "failed");
  }

  botStarted() {
    this.botTalking = true;
    if (!this.speaking && this.toSpeak.length) this.startSpeaking(this.toSpeak[0]);
    this.changed();
  }

  botStopped() {
    this.botTalking = false;
    this.botQuietAt = this.now();
    // OV speaks an answer a sentence at a time and stops after each.
    if (this.speaking?.lastSentence) this.endSpeaking("spoken");
    this.changed();
  }

  // The text of each sentence OV speaks comes back, which tells whose answer is
  // playing; an answer OV gives on its own, to no question, still matches its
  // response. `played` is true for the TTS text, false for the bot's output text.
  botSaid(text: string, played: boolean) {
    const said = norm(text);
    if (!said) return;
    const owner = [this.speaking, ...this.toSpeak].find((r) => r && r.textNorm.includes(said)) || null;
    if (!owner) {
      if (this.speaking && !this.speaking.confirmed) this.putBack(this.speaking);
      this.changed();
      return;
    }
    if (owner !== this.speaking) {
      if (this.speaking?.confirmed) this.endSpeaking("spoken");
      else if (this.speaking) this.putBack(this.speaking);
      this.startSpeaking(owner);
    }
    owner.confirmed = true;
    if (played && owner.textNorm.endsWith(said)) owner.lastSentence = true;
    this.changed();
  }

  /** Call every half second or so: ends answers and flags stalled turns. */
  tick() {
    const now = this.now();
    let dirty = false;
    if (this.speaking && !this.botTalking && now - this.botQuietAt > ANSWER_QUIET_MS) {
      this.endSpeaking("spoken");
      dirty = true;
    }
    // A written answer waits while the user talks or OV says something else. One
    // that waits longer was dropped: the bot drops the answers waiting whenever
    // its VAD hears the user, which a pause inside a turn does not show here.
    for (const r of this.toSpeak) if (this.serverHearing || this.botTalking || this.speaking) r.readyAt = now;
    for (const r of this.toSpeak.filter((x) => now - (x.readyAt ?? now) > ANSWER_WAIT_MS)) {
      this.toSpeak = this.toSpeak.filter((x) => x !== r);
      this.settle(r, "dropped");
      dirty = true;
    }
    for (const turn of this.turns) {
      if (turn.final || turn.warned) continue;
      const stall = this.stalled(turn, now);
      if (stall) {
        this.update(turn, { warned: true, ...stall });
        dirty = true;
      }
    }
    if (dirty) this.changed();
  }

  /** Forget the session, for a new connection. */
  reset() {
    this.turns = [];
    this.turnByInput.clear();
    this.responses.clear();
    this.earlyInputs.clear();
    this.toSpeak = [];
    this.speaking = null;
    this.notes = [];
    this.stats = { inputs: 0, answers: 0, latencies: [], strip: [], lastInputDoneAt: null };
    this.serverHearing = false;
    this.botTalking = false;
    this.botQuietAt = -Infinity;
    this.botReady = false;
    this.changed();
  }

  // ---- Helpers -----------------------------------------------------------------
  private newest(test: (t: Turn) => boolean) {
    for (let i = this.turns.length - 1; i >= 0; i--) if (test(this.turns[i])) return this.turns[i];
    return undefined;
  }

  private newTurn(fields: Partial<Turn> = {}): Turn {
    const turn: Turn = {
      id: this.nextTurnId++,
      t0: this.now(),
      ended: null,
      sentAt: null,
      finishedAt: null,
      text: "",
      segments: 1,
      inputId: null,
      responseId: null,
      step: 0,
      tone: "you",
      status: "检测到你在说话",
      note: "",
      final: false,
      warned: false,
      ...fields,
    };
    this.turns.push(turn);
    while (this.turns.length > KEEP_TURNS) this.dropTurn(this.turns[0]);
    return turn;
  }

  private dropTurn(turn: Turn) {
    this.turns = this.turns.filter((t) => t !== turn);
    if (turn.inputId) this.turnByInput.delete(turn.inputId);
  }

  private update(turn: Turn, fields: Partial<Turn>) {
    Object.assign(turn, fields);
    if ("final" in fields) turn.finishedAt = fields.final ? this.now() : null;
  }

  private turnOf(r: OvResponse) {
    return r.turnId == null ? undefined : this.turns.find((t) => t.id === r.turnId);
  }

  private settle(r: OvResponse, state: ResponseState) {
    r.state = state;
    const turn = this.turnOf(r);
    if (!turn) return;
    const how: Partial<Record<ResponseState, Partial<Turn>>> = {
      spoken: { status: "OV 说完了" },
      interrupted: { status: "OV 说到一半被你打断" },
      dropped: { tone: "silent", status: "回答没有播放：你又开始说话了" },
      superseded: { tone: "silent", status: "被 OV 更新的回答取代" },
      cancelled: { tone: "warn", status: "回答被取消", note: r.reason ? `原因：${r.reason}` : "" },
      failed: { tone: "warn", status: "回答失败", note: r.reason ? `原因：${r.reason}` : "" },
    };
    this.update(turn, { ...how[state], final: true });
  }

  private startSpeaking(r: OvResponse) {
    this.toSpeak = this.toSpeak.filter((x) => x !== r);
    this.speaking = r;
    r.state = "speaking";
    r.confirmed = false;
    r.lastSentence = false;
    const turn = this.turnOf(r);
    if (turn) this.update(turn, { step: 4, status: "OV 正在说", note: "" });
  }

  private putBack(r: OvResponse) {
    this.speaking = null;
    r.state = "waiting";
    r.readyAt = this.now();
    this.toSpeak.unshift(r);
    const turn = this.turnOf(r);
    if (turn) this.update(turn, { step: 3, status: "回答写好了，等待播放" });
  }

  private endSpeaking(state: "spoken" | "interrupted" | "superseded") {
    const r = this.speaking;
    this.speaking = null;
    if (r) this.settle(r, state);
  }

  private stalled(turn: Turn, now: number): Partial<Turn> | null {
    if (turn.ended == null && !turn.inputId && now - turn.t0 > LONG_TURN_MS) {
      return {
        note: `服务器认为你已经说了 ${Math.round((now - turn.t0) / 1000)} 秒还没说完：背景噪音会让它一直等下去`,
      };
    }
    if (turn.ended != null && !turn.inputId && now - turn.ended > QUESTION_WAIT_MS) {
      return turn.text
        ? { tone: "warn", status: "识别到了，但还没发给 OV", note: "可能会和你的下一句合在一起发出" }
        : {
            tone: "warn",
            status: "没有识别出文字",
            note: "ASR 没有返回文字（太短、太轻或只是噪音），这句不会发给 OV",
            final: true,
          };
    }
    if (turn.step === 2 && turn.sentAt != null && now - turn.sentAt > DECIDE_WAIT_MS) {
      return { tone: "warn", note: `发出 ${secs(now - turn.sentAt)} 了，OV 还没处理这一帧：OV 可能在排队或卡住了` };
    }
    return null;
  }

  private trimResponses() {
    while (this.responses.size > KEEP_TURNS * 2) this.responses.delete(this.responses.keys().next().value!);
  }

  private note(kind: Note["kind"], text: string) {
    this.notes.push({ id: `${kind}-${this.now()}-${this.notes.length}`, at: this.now(), kind, text });
    if (this.notes.length > 20) this.notes.shift();
  }
}
