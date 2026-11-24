import {
  RTVIEvent,
  type BotOutputData,
  type BotTTSTextData,
  type PipecatClient,
  type TranscriptData,
} from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { OvSession, type Snapshot } from "./session";

/** The session model, fed from the client's RTVI events. */
export function useOvSession(): { session: OvSession; snap: Snapshot } {
  const session = useMemo(() => new OvSession(), []);
  const snap = useSyncExternalStore(session.subscribe, session.getSnapshot);

  useRTVIClientEvent(RTVIEvent.Connected, useCallback(() => session.reset(), [session]));
  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback((d: any) => session.onServerMessage(d?.ov ? d : d?.data), [session]),
  );
  useRTVIClientEvent(RTVIEvent.UserStartedSpeaking, useCallback(() => session.userStarted(), [session]));
  useRTVIClientEvent(RTVIEvent.UserStoppedSpeaking, useCallback(() => session.userStopped(), [session]));
  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    useCallback((d: TranscriptData) => session.userTranscript(d.text, d.final !== false), [session]),
  );
  useRTVIClientEvent(RTVIEvent.BotStartedSpeaking, useCallback(() => session.botStarted(), [session]));
  useRTVIClientEvent(RTVIEvent.BotStoppedSpeaking, useCallback(() => session.botStopped(), [session]));
  useRTVIClientEvent(RTVIEvent.BotTtsText, useCallback((d: BotTTSTextData) => session.botSaid(d.text, true), [session]));
  useRTVIClientEvent(RTVIEvent.BotOutput, useCallback((d: BotOutputData) => session.botSaid(d.text, false), [session]));

  useEffect(() => {
    const id = setInterval(() => session.tick(), 500);
    return () => clearInterval(id);
  }, [session]);

  return { session, snap };
}

// ---- Is the server hearing the microphone? ----------------------------------------
// The local level is Daily's (0..1, ten times a second). Voice is a stretch of
// loud readings with short gaps, loud often enough to be speech rather than keys
// or knocks. While OV talks the microphone may carry its echo, so then nothing
// counts. The stretch restarts when the server ends a turn: the voice it heard is
// not owed a report, only the voice after it.
const VOICE_LEVEL = 0.05;
const VOICE_GAP_MS = 600;
const VOICE_SHARE = 0.3;
const ECHO_MS = 1000;
const NOT_HEARD_MS = 2000;
const NOT_HEARD_HOLD_MS = 4000;
const SILENT_MIC_MS = 4000; // connected, mic on, and the browser's level never above zero

export interface Hearing {
  level: number;
  notHeard: boolean; // the mic hears voice the server never reports
  micSilent: boolean; // the browser hands the page a silent microphone
}

export function useHearing(
  connected: boolean,
  micEnabled: boolean,
  session: OvSession,
): Hearing {
  const [level, setLevel] = useState(0);
  const [notHeard, setNotHeard] = useState(false);
  const [micSilent, setMicSilent] = useState(false);
  const st = useRef({ since: 0, last: -Infinity, ms: 0, prev: 0, wasHearing: false, loudestAt: -Infinity, connectedAt: 0 });

  useEffect(() => {
    st.current.connectedAt = performance.now();
    st.current.loudestAt = -Infinity;
    setNotHeard(false);
    setMicSilent(false);
  }, [connected]);

  useRTVIClientEvent(
    RTVIEvent.LocalAudioLevel,
    useCallback(
      (lv: number) => {
        const now = performance.now();
        const s = st.current;
        setLevel(lv);
        if (lv > 0.001) s.loudestAt = now;
        if (!connected) return;
        setMicSilent(micEnabled && now - s.connectedAt > SILENT_MIC_MS && now - s.loudestAt > SILENT_MIC_MS);

        const frameMs = s.prev ? Math.min(200, now - s.prev) : 0;
        s.prev = now;
        if (s.wasHearing && !session.serverHearing) {
          s.since = now; // the server just ended a turn
          s.ms = 0;
        }
        s.wasHearing = session.serverHearing;
        const echo = session.botTalking || now - session.botQuietAt < ECHO_MS;
        if (micEnabled && !echo && lv >= VOICE_LEVEL) {
          if (now - s.last > VOICE_GAP_MS) {
            s.since = now;
            s.ms = 0;
          }
          s.last = now;
          s.ms += frameMs;
        }
        setNotHeard((shown) => {
          if (shown) return !(session.serverHearing || now - s.last > NOT_HEARD_HOLD_MS);
          const span = now - s.since;
          const talking = now - s.last <= VOICE_GAP_MS && span > NOT_HEARD_MS && s.ms > VOICE_SHARE * span;
          return talking && session.botReady && !session.serverHearing;
        });
      },
      [connected, micEnabled, session],
    ),
  );
  return { level, notHeard, micSilent };
}

// ---- Connection health, from the browser's WebRTC stats -----------------------------
export interface Health {
  path: string;
  relay: boolean;
  rttMs: number | null;
  audioKbps: number | null;
  videoKbps: number | null;
  fps: number | null;
  lossPct: number | null;
}

const EMPTY_HEALTH: Health = { path: "", relay: false, rttMs: null, audioKbps: null, videoKbps: null, fps: null, lossPct: null };

/** The SmallWebRTC transport's peer connection; it is not part of the public API. */
export function peerConnection(client: PipecatClient | null): RTCPeerConnection | null {
  const transport = client?.transport as unknown as { pc?: RTCPeerConnection | null } | undefined;
  return transport?.pc ?? null;
}

export function useHealth(client: PipecatClient | null, connected: boolean): Health {
  const [health, setHealth] = useState<Health>(EMPTY_HEALTH);
  useEffect(() => {
    if (!connected) {
      setHealth(EMPTY_HEALTH);
      return;
    }
    const prev: Record<string, { bytes: number; at: number }> = {};
    const rate = (key: string, out?: any) => {
      if (!out) return null;
      const p = prev[key];
      prev[key] = { bytes: out.bytesSent, at: out.timestamp };
      return p && out.timestamp > p.at ? ((out.bytesSent - p.bytes) * 8) / (out.timestamp - p.at) : null;
    };
    const refresh = async () => {
      const pc = peerConnection(client);
      if (!pc) return;
      let report: RTCStatsReport;
      try {
        report = await pc.getStats();
      } catch {
        return;
      }
      const stats = [...report.values()] as any[];
      const transport = stats.find((s) => s.type === "transport" && s.selectedCandidatePairId);
      const pair = transport
        ? report.get(transport.selectedCandidatePairId)
        : stats.find((s) => s.type === "candidate-pair" && s.nominated && s.state === "succeeded");
      const near = pair && report.get(pair.localCandidateId);
      // The camera is the transceiver with a live video track; the screen one has none.
      const outbound = stats.filter((s) => s.type === "outbound-rtp");
      const audio = outbound.find((s) => s.kind === "audio");
      const video = outbound
        .filter((s) => s.kind === "video")
        .sort((a, b) => (b.bytesSent || 0) - (a.bytesSent || 0))[0];
      const loss = [audio, video]
        .map((o) => o && stats.find((s) => s.type === "remote-inbound-rtp" && s.localId === o.id))
        .map((b) => (b && b.fractionLost != null ? b.fractionLost * 100 : null))
        .filter((x): x is number => x != null);
      setHealth({
        path: near ? `${near.candidateType}/${near.protocol}${near.relayProtocol ? ` · TURN ${near.relayProtocol}` : ""}` : "连接中…",
        relay: near?.candidateType === "relay",
        rttMs: pair?.currentRoundTripTime != null ? Math.round(pair.currentRoundTripTime * 1000) : null,
        audioKbps: rate("audio", audio),
        videoKbps: rate("video", video),
        fps: video?.framesPerSecond ?? null,
        lossPct: loss.length ? Math.max(...loss) : null,
      });
    };
    refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [client, connected]);
  return health;
}

// ---- Keep the camera light ---------------------------------------------------------
// OV takes one frame a second, so the camera need not be sharp or fast. A
// full-rate camera fills a slow or TCP-only path (a VPN plus TURN over TCP) and
// the audio behind it never reaches the bot. ?vbr=<kbps>&vfps=<n> override.
const query = new URLSearchParams(location.search);
export const VIDEO_KBPS = Number(query.get("vbr")) || 250;
export const VIDEO_FPS = Number(query.get("vfps")) || 5;

export function useLightCamera(client: PipecatClient | null) {
  useRTVIClientEvent(
    RTVIEvent.TrackStarted,
    useCallback((track: MediaStreamTrack, participant?: { local: boolean }) => {
      if (participant?.local && track.kind === "video") {
        track
          .applyConstraints({ width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: VIDEO_FPS, max: VIDEO_FPS } })
          .catch(() => {});
      }
    }, []),
  );
  useRTVIClientEvent(
    RTVIEvent.Connected,
    useCallback(() => {
      const pc = peerConnection(client);
      if (!pc) return;
      for (const sender of pc.getSenders()) {
        const kind = sender.track?.kind;
        if (!kind) continue;
        const params = sender.getParameters();
        if (!params.encodings?.length) continue;
        params.encodings = params.encodings.map((e) =>
          kind === "video"
            ? { ...e, maxBitrate: VIDEO_KBPS * 1000, maxFramerate: VIDEO_FPS, priority: "low", networkPriority: "low" }
            : { ...e, priority: "high", networkPriority: "high" },
        );
        sender.setParameters(params).catch(() => {});
      }
    }, [client]),
  );
}
