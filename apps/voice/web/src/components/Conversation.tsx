import { useEffect, useRef } from "react";

import { STEPS, type OvResponse, type ResponseState, type Snapshot, type Tone, type Turn } from "../ov/session";

const TONE_TEXT: Record<Tone, string> = {
  you: "text-you",
  ov: "text-ov",
  caption: "text-ov",
  silent: "text-dim",
  warn: "text-caution",
};

const RESPONSE_STATE: Record<ResponseState, { label: string; className: string }> = {
  writing: { label: "生成中…", className: "text-dim" },
  waiting: { label: "等待播放", className: "text-dim" },
  speaking: { label: "正在说", className: "text-ov" },
  spoken: { label: "说完了", className: "text-dim" },
  interrupted: { label: "被你打断", className: "text-caution" },
  dropped: { label: "没有播放：你又开始说话了", className: "text-dim" },
  superseded: { label: "被新回答取代", className: "text-dim" },
  cancelled: { label: "已取消", className: "text-caution" },
  failed: { label: "失败", className: "text-bad" },
  caption: { label: "解说（不念出来）", className: "text-dim" },
  empty: { label: "空的", className: "text-dim" },
};

function Steps({ turn }: { turn: Turn }) {
  return (
    <div className="mt-2" aria-label={`进度：${STEPS.slice(0, turn.step + 1).join(" → ")}`}>
      <div className="grid grid-cols-5 gap-1">
        {STEPS.map((name, i) => {
          const done = i < turn.step || (i === turn.step && turn.final && turn.tone !== "warn");
          const current = i === turn.step && !turn.final;
          const failed = i === turn.step && turn.tone === "warn";
          return (
            <div key={name} className="flex flex-col gap-1">
              <span
                className={[
                  "h-1 rounded-full",
                  failed ? "bg-caution" : done ? "bg-you" : current ? "bg-you/50 animate-pulse" : "bg-raised",
                ].join(" ")}
              />
              <span className={`text-[10px] ${i <= turn.step ? "text-dim" : "text-dim/50"}`}>{name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function OvBubble({ response, nested }: { response: OvResponse; nested?: boolean }) {
  const caption = response.control === "caption";
  const state = RESPONSE_STATE[response.state];
  return (
    <div className={`${nested ? "mt-3" : ""} rounded-lg border border-line bg-raised/40 px-3 py-2`}>
      <div className="flex items-center gap-2 text-xs">
        <span className="font-semibold text-ov">{caption ? "OV 解说" : "OV"}</span>
        <span className={state.className}>{state.label}</span>
        {response.reason && <span className="text-dim">· {response.reason}</span>}
      </div>
      <p className="mt-1 whitespace-pre-wrap break-words text-sm text-ink">
        {response.text || <span className="italic text-dim">…</span>}
      </p>
    </div>
  );
}

function TurnCard({ turn, response }: { turn: Turn; response?: OvResponse }) {
  return (
    <article className="rounded-xl border border-line bg-surface px-4 py-3">
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <span className="font-semibold text-you">你</span>
        <span className={`truncate ${TONE_TEXT[turn.tone]}`}>{turn.status}</span>
      </div>
      <p className="mt-1 whitespace-pre-wrap break-words text-[15px] text-ink">
        {turn.text || <span className="italic text-dim">（还没识别出文字）</span>}
      </p>
      <Steps turn={turn} />
      {turn.note && <p className="mt-2 text-xs text-dim">{turn.note}</p>}
      {response && <OvBubble response={response} nested />}
    </article>
  );
}

type Entry = { at: number; turn?: Turn; response?: OvResponse };

/** Captions show only from `captionsSince` on, and not at all when it is null. */
export function Conversation({ snap, captionsSince }: { snap: Snapshot; captionsSince: number | null }) {
  const shown = snap.responses.filter(
    (r) => r.control !== "caption" || (captionsSince != null && r.createdAt >= captionsSince),
  );
  const byId = new Map(shown.map((r) => [r.id, r]));
  const entries: Entry[] = [
    ...snap.turns.map((turn) => ({ at: turn.t0, turn, response: turn.responseId ? byId.get(turn.responseId) : undefined })),
    ...shown.filter((r) => r.turnId == null).map((response) => ({ at: response.createdAt, response })),
  ].sort((a, b) => a.at - b.at);

  const end = useRef<HTMLDivElement>(null);
  const last = entries.at(-1);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [entries.length, last?.turn?.status, last?.response?.state, last?.response?.text]);

  if (!entries.length) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-sm text-dim">
        <p>对着摄像头说一句话，或者在下面打字。</p>
        <p>每句话会一步步显示：服务器听到 → 识别成文字 → 发给 OV → OV 的决定 → OV 开口。</p>
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-3 p-4">
      {entries.map((e) =>
        e.turn ? (
          <TurnCard key={`t${e.turn.id}`} turn={e.turn} response={e.response} />
        ) : (
          <OvBubble key={`r${e.response!.id}`} response={e.response!} />
        ),
      )}
      <div ref={end} />
    </div>
  );
}
