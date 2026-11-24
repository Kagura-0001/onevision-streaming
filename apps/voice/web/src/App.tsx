import type { PipecatClient } from "@pipecat-ai/client-js";
import {
  PipecatClientVideo,
  usePipecatClientMicControl,
  usePipecatClientTransportState,
} from "@pipecat-ai/client-react";
import {
  Badge,
  Banner,
  BannerTitle,
  Button,
  PipecatAppBase,
  TextInput,
  UserAudioControl,
  UserVideoControl,
  VoiceVisualizer,
} from "@pipecat-ai/voice-ui-kit";
import { useCallback, useEffect, useState } from "react";

import { Conversation } from "./components/Conversation";
import { DevPanel } from "./components/DevPanel";
import { useHealth, useHearing, useLightCamera, useOvSession } from "./ov/hooks";
import type { Snapshot } from "./ov/session";

// The bot serves this page at /live/ and its routes one level up; the transport
// turns ".../start" into ".../sessions/<id>/api/offer" for the WebRTC offer.
const START_URL = new URL("../start", location.href).href;

const START_BOT_PARAMS = { endpoint: START_URL, requestData: { transport: "webrtc" } };

export default function App() {
  return (
    <PipecatAppBase
      transportType="smallwebrtc"
      startBotParams={START_BOT_PARAMS}
      clientOptions={{ enableMic: true, enableCam: true }}
      themeProps={{ defaultTheme: "dark" }}
      initDevicesOnMount
    >
      {({ client, handleConnect, handleDisconnect, error }) =>
        // The client, and the providers the kit's components need, exist only once it is made.
        client ? (
          <Live
            client={client}
            connect={handleConnect}
            disconnect={handleDisconnect}
            error={error ?? null}
          />
        ) : (
          <div className="grid h-full place-items-center text-sm text-dim">正在准备…</div>
        )
      }
    </PipecatAppBase>
  );
}

interface LiveProps {
  client: PipecatClient;
  connect?: () => void | Promise<void>;
  disconnect?: () => void | Promise<void>;
  error: string | null;
}

function Live({ client, connect, disconnect, error }: LiveProps) {
  const state = usePipecatClientTransportState();
  const connected = state === "connected" || state === "ready";
  const busy = ["authenticating", "authenticated", "connecting"].includes(state);
  const { enableMic, isMicEnabled } = usePipecatClientMicControl();
  const { session, snap } = useOvSession();
  const health = useHealth(client, connected);
  useLightCamera(client);

  const [pushToTalk, setPushToTalk] = useState(false);
  // OV always writes captions; the page shows the ones written since the switch went on.
  const [captionsSince, setCaptionsSince] = useState<number | null>(null);
  const hearing = useHearing(connected, isMicEnabled && !pushToTalk, session);
  const [showDev, setShowDev] = useState(false);
  const [pane, setPane] = useState<"camera" | "chat">("chat");
  const [model, setModel] = useState("");

  useEffect(() => {
    fetch(new URL("../model", location.href))
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => m?.model && setModel(m.model))
      .catch(() => {});
  }, []);

  // Push to talk: the microphone sends silence unless the button or space is held.
  const talk = useCallback((on: boolean) => enableMic(on), [enableMic]);
  useEffect(() => {
    if (connected) enableMic(!pushToTalk);
  }, [connected, pushToTalk, enableMic]);
  useEffect(() => {
    if (!pushToTalk) return;
    const isTyping = (e: KeyboardEvent) => e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat && !isTyping(e)) {
        e.preventDefault();
        talk(true);
      }
    };
    const up = (e: KeyboardEvent) => e.code === "Space" && !isTyping(e) && talk(false);
    addEventListener("keydown", down);
    addEventListener("keyup", up);
    return () => {
      removeEventListener("keydown", down);
      removeEventListener("keyup", up);
    };
  }, [pushToTalk, talk]);

  const sendText = useCallback(
    (text: string) => {
      text = text.trim();
      if (!text || !client || !connected) return;
      client.sendClientMessage("ov-text", { text });
      session.typed(text);
    },
    [client, connected, session],
  );

  return (
    <div className="flex h-full flex-col bg-page text-ink">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-ov font-mono text-xs font-bold text-white">OV</span>
          <span className="font-semibold">OV Live</span>
          {model && <span className="hidden font-mono text-xs text-dim sm:inline">{model}</span>}
        </div>
        <ConnectionBadge state={state} />
        <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
          <Toggle
            label="画面解说"
            hint="打开后，对话里显示 OV 从这一刻起写的画面描述，不念出来"
            checked={captionsSince != null}
            onChange={(on) => setCaptionsSince(on ? performance.now() : null)}
          />
          <Toggle label="按住说话" hint="开启后麦克风平时静音，按住按钮或空格键时才收音" checked={pushToTalk} onChange={setPushToTalk} />
          <Button size="sm" variant={showDev ? "secondary" : "ghost"} onClick={() => setShowDev((v) => !v)}>
            开发者
          </Button>
          {connected || busy ? (
            <Button size="sm" variant="outline" onClick={() => disconnect?.()} disabled={!connected}>
              结束
            </Button>
          ) : (
            <Button size="sm" variant="primary" onClick={() => connect?.()}>
              开始
            </Button>
          )}
        </div>
      </header>

      <nav className="flex gap-1 border-b border-line px-4 py-1.5 lg:hidden" aria-label="切换视图">
        {(["camera", "chat"] as const).map((p) => (
          <Button key={p} size="sm" variant={pane === p ? "secondary" : "ghost"} onClick={() => setPane(p)}>
            {p === "camera" ? "画面" : "对话"}
          </Button>
        ))}
      </nav>

      <main className="grid min-h-0 flex-1 gap-4 p-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <section className={`${pane === "camera" ? "flex" : "hidden"} min-h-0 flex-col gap-4 overflow-y-auto lg:flex`}>
          <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-line bg-black">
            {connected ? (
              <PipecatClientVideo participant="local" fit="cover" mirror className="h-full w-full" />
            ) : (
              <div className="absolute inset-0 grid place-items-center px-8 text-center text-sm text-dim">
                点“开始”后，浏览器会请求麦克风和摄像头。画面每秒送给 OV 一帧，你说完一句话，问题随下一帧发出。
              </div>
            )}
          </div>
          <AgentState snap={snap} connected={connected} />
          <div className="flex flex-wrap items-center gap-3">
            <UserAudioControl size="md" />
            <UserVideoControl size="md" noVideo />
            <MicLevel level={hearing.level} on={isMicEnabled} />
          </div>
          <Warnings hearing={hearing} error={error} connected={connected} />
        </section>

        <section className={`${pane === "chat" ? "flex" : "hidden"} min-h-0 flex-col overflow-hidden rounded-xl border border-line bg-surface/40 lg:flex`}>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {showDev ? <DevPanel snap={snap} health={health} connected={connected} /> : <Conversation snap={snap} captionsSince={captionsSince} />}
          </div>
          <div className="flex items-center gap-2 border-t border-line p-3">
            {pushToTalk && connected && <HoldToTalk onChange={talk} active={isMicEnabled} />}
            <div className="min-w-0 flex-1">
              <TextInput
                onSend={sendText}
                disabled={!connected}
                placeholder={connected ? "打字问 OV，回车发送" : "连接后可以打字提问"}
                buttonLabel="发送"
              />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function ConnectionBadge({ state }: { state: string }) {
  const map: Record<string, { text: string; color: "active" | "warning" | "destructive" | "inactive" }> = {
    connected: { text: "已连接", color: "active" },
    ready: { text: "已连接", color: "active" },
    error: { text: "出错", color: "destructive" },
    disconnected: { text: "未连接", color: "inactive" },
    initializing: { text: "未连接", color: "inactive" },
    initialized: { text: "未连接", color: "inactive" },
    disconnecting: { text: "断开中", color: "inactive" },
  };
  const s = map[state] ?? { text: "连接中…", color: "warning" as const };
  return (
    <Badge size="sm" variant="filled" color={s.color}>
      {s.text}
    </Badge>
  );
}

function Toggle(props: { label: string; hint: string; checked: boolean; disabled?: boolean; onChange: (on: boolean) => void }) {
  return (
    <label className={`flex items-center gap-1.5 ${props.disabled ? "opacity-50" : "cursor-pointer"}`} title={props.hint}>
      <input
        type="checkbox"
        className="accent-[var(--color-agent)]"
        checked={props.checked}
        disabled={props.disabled}
        onChange={(e) => props.onChange(e.target.checked)}
      />
      {props.label}
    </label>
  );
}

/** What OV is doing now, the one thing to glance at. */
function AgentState({ snap, connected }: { snap: Snapshot; connected: boolean }) {
  const thinking = snap.responses.some((r) => r.state === "writing" && r.control === "response");
  const [title, sub] = !connected
    ? ["未连接", "点“开始”和 OV 对话"]
    : snap.botTalking || snap.speakingResponseId
      ? ["OV 在说", "开口说话会打断它"]
      : snap.serverHearing
        ? ["在听你说", "说完停一下，OV 就会收到"]
        : thinking
          ? ["OV 在想…", "回答写好就会念出来"]
          : ["等你开口", "OV 一直在看画面，也可能主动说话"];
  return (
    <div className="flex items-center gap-4 rounded-xl border border-line bg-surface px-4 py-3">
      <div className="min-w-0 flex-1">
        <div className={`text-xl font-semibold ${snap.botTalking ? "text-ov" : snap.serverHearing ? "text-you" : "text-ink"}`}>
          {title}
        </div>
        <div className="text-xs text-dim">{sub}</div>
      </div>
      <div className="h-12 w-40 shrink-0">
        <VoiceVisualizer participantType="bot" barCount={12} barWidth={6} barGap={4} barMaxHeight={44} barColor="var(--color-agent)" />
      </div>
    </div>
  );
}

function MicLevel({ level, on }: { level: number; on: boolean }) {
  return (
    <div className="flex min-w-32 flex-1 items-center gap-2 text-xs text-dim" title="浏览器拿到的麦克风音量">
      <span>{on ? "麦克风" : "已静音"}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-raised">
        <div className="h-full bg-you transition-[width] duration-100" style={{ width: `${Math.min(100, level * 300)}%` }} />
      </div>
    </div>
  );
}

function HoldToTalk({ onChange, active }: { onChange: (on: boolean) => void; active: boolean }) {
  return (
    <Button
      size="md"
      variant={active ? "active" : "outline"}
      className="touch-none select-none"
      onPointerDown={(e) => {
        e.preventDefault();
        onChange(true);
      }}
      onPointerUp={() => onChange(false)}
      onPointerLeave={() => onChange(false)}
      onPointerCancel={() => onChange(false)}
    >
      {active ? "正在收音…" : "按住说话"}
    </Button>
  );
}

function Warnings({ hearing, error, connected }: { hearing: { notHeard: boolean; micSilent: boolean }; error: string | null; connected: boolean }) {
  return (
    <div className="flex flex-col gap-2">
      {error && (
        <Banner variant="destructive">
          <BannerTitle>{error}</BannerTitle>
        </Banner>
      )}
      {connected && hearing.micSilent && (
        <Banner variant="destructive">
          <BannerTitle>
            浏览器给的麦克风一点声音都没有。macOS 上通常是系统没给浏览器麦克风权限：系统设置 → 隐私与安全性 → 麦克风，打开你的浏览器，然后完全退出浏览器再打开。也可以在上面换一个麦克风试试。
          </BannerTitle>
        </Banner>
      )}
      {connected && hearing.notHeard && !hearing.micSilent && (
        <Banner variant="primary">
          <BannerTitle>
            麦克风有声音，但 2 秒多了服务器还没听到你在说话：多半是网络慢或音频堵在路上。可以打开“开发者”看连接，或者先打字提问。
          </BannerTitle>
        </Banner>
      )}
    </div>
  );
}
