import { EventsPanel } from "@pipecat-ai/voice-ui-kit";

import type { Health } from "../ov/hooks";
import type { Snapshot } from "../ov/session";

const STRIP_COLOR: Record<string, string> = { response: "bg-ov", caption: "bg-you", silence: "bg-raised" };

function Stat({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="rounded-lg bg-raised/60 px-3 py-2">
      <div className="font-mono text-base tabular-nums text-ink">{value}</div>
      <div className="text-[11px] text-dim">{label}</div>
    </div>
  );
}

function Row({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="flex justify-between gap-4 text-xs">
      <span className="text-dim">{label}</span>
      <span className={`font-mono tabular-nums ${warn ? "text-caution" : "text-ink"}`}>{value}</span>
    </div>
  );
}

/** What we look at when something is off; folded away by default. */
export function DevPanel({ snap, health, connected }: { snap: Snapshot; health: Health; connected: boolean }) {
  const { latencies, inputs, answers, strip, lastInputDoneAt } = snap.stats;
  const last = latencies.at(-1);
  const avg = latencies.length ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : null;
  const since = lastInputDoneAt != null ? (performance.now() - lastInputDoneAt) / 1000 : null;
  const kbps = (v: number | null) => (v == null ? "–" : `${Math.round(v)} kbps`);
  return (
    <div className="flex flex-col gap-4 p-4">
      <section className="grid grid-cols-4 gap-2">
        <Stat value={last ?? "–"} label="上一帧延迟 ms" />
        <Stat value={avg ?? "–"} label="平均 ms" />
        <Stat value={inputs} label="输入数" />
        <Stat value={answers} label="回答数" />
      </section>

      <section>
        <div className="mb-1 flex justify-between text-[11px] text-dim">
          <span>OV 在最近 60 帧上的决策</span>
          <span>
            <i className="mr-1 inline-block h-2 w-2 rounded-sm bg-raised" />不出声
            <i className="ml-3 mr-1 inline-block h-2 w-2 rounded-sm bg-ov" />回答
            <i className="ml-3 mr-1 inline-block h-2 w-2 rounded-sm bg-you" />解说
          </span>
        </div>
        <div className="flex h-4 gap-px" aria-label="OV 的决策">
          {Array.from({ length: 60 }, (_, i) => strip[strip.length - 60 + i]).map((c, i) => (
            <i key={i} className={`flex-1 rounded-[2px] ${c ? STRIP_COLOR[c] ?? "bg-raised" : "border border-dashed border-line"}`} />
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-1.5">
        <h3 className="text-xs font-semibold text-ink">连接</h3>
        <Row label="路径" value={connected ? health.path || "–" : "未连接"} />
        <Row label="往返延迟" value={health.rttMs != null ? `${health.rttMs} ms` : "–"} warn={(health.rttMs ?? 0) > 400} />
        <Row label="音频上行" value={kbps(health.audioKbps)} warn={connected && health.audioKbps != null && health.audioKbps < 2} />
        <Row
          label="画面上行"
          value={`${kbps(health.videoKbps)}${health.fps ? ` · ${Math.round(health.fps)} fps` : ""}`}
          warn={connected && health.videoKbps != null && health.videoKbps < 10}
        />
        <Row label="丢包" value={health.lossPct != null ? `${health.lossPct.toFixed(0)}%` : "–"} warn={(health.lossPct ?? 0) > 5} />
        <Row
          label="OV 最近收到画面"
          value={since != null ? `${since.toFixed(0)} s 前` : "–"}
          warn={connected && since != null && since > 5}
        />
      </section>

      {snap.notes.length > 0 && (
        <section className="flex flex-col gap-1">
          <h3 className="text-xs font-semibold text-ink">会话记录</h3>
          {snap.notes.map((n) => (
            <p key={n.id} className={`text-xs ${n.kind === "error" ? "text-bad" : "text-dim"}`}>
              {n.text}
            </p>
          ))}
        </section>
      )}

      <section className="h-72 min-h-0 overflow-hidden rounded-lg border border-line">
        <EventsPanel />
      </section>
    </div>
  );
}
