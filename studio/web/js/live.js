// Live streams in the pod mode (docs/pod-mode.md phase 2): WHEP playback of
// the pod's Cloudflare Stream output, and the stream's state line.

export const LIVE_TERMINAL = new Set(["closed", "ended", "failed", "stopped"]);

let pc = null, resource = null;

// Play a WHEP URL into `video`: offer recvonly video + audio, POST the SDP,
// set the answer; the Location header is the resource to DELETE on stop.
export async function playWhep(video, url) {
  await stopPlayer(video);
  pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.cloudflare.com:3478" }] });
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });
  pc.ontrack = (ev) => { video.srcObject = ev.streams[0]; video.play?.().catch(() => {}); };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await new Promise((res) => {
    if (pc.iceGatheringState === "complete") return res();
    const t = setTimeout(res, 2000);
    pc.onicegatheringstatechange = () => { if (pc.iceGatheringState === "complete") { clearTimeout(t); res(); } };
  });
  const r = await fetch(url, { method: "POST", headers: { "content-type": "application/sdp" }, body: pc.localDescription.sdp });
  if (!r.ok) throw new Error(`WHEP answered ${r.status}`);
  const loc = r.headers.get("location");
  resource = loc ? new URL(loc, url).href : null;
  await pc.setRemoteDescription({ type: "answer", sdp: await r.text() });
}

export async function stopPlayer(video) {
  if (resource) { try { await fetch(resource, { method: "DELETE" }); } catch {} resource = null; }
  if (pc) { pc.close(); pc = null; }
  if (video) video.srcObject = null;
}

export function playing() { return !!pc; }

// One line for the stream's state.
export function liveText(s) {
  if (!s) return "";
  const st = s.status || {}, ses = s.session || {};
  const parts = [`stream ${s.id}`, `${s.mode || "?"}`, st.state || "?", `output ${st.output_state || "?"}`];
  if (st.output_error) parts.push(`(${st.output_error})`);
  if (s.mode === "clip") parts.push(`${ses.playing ? "playing" : "idle"}, ${ses.generation_queued ?? "?"} generating, ${ses.playout_queued ?? "?"} queued, autoplay ${ses.autoplay ?? "?"}`);
  else if (ses.prompt) parts.push(`prompt: ${String(ses.prompt).slice(0, 60)}`);
  if (st.end_reason) parts.push(st.end_reason);
  return parts.join(" · ");
}

export function isLiveTerminal(s) {
  const st = s?.status || {};
  return LIVE_TERMINAL.has(st.state) || st.output_state === "failed" || st.output_state === "ended";
}

// The commands by mode: clip streams enqueue clips and toggle autoplay,
// causal ones switch the prompt at the next block and pause the rollout.
export function commandFor(mode, action, data) {
  if (action === "steer") return mode === "clip" ? { type: "enqueue", data: data.seconds > 0 ? { prompt: data.prompt, seconds: data.seconds } : { prompt: data.prompt } } : { type: "set_prompt", data: { prompt: data.prompt } };
  if (action === "pause") return mode === "clip" ? { type: "set_autoplay", data: false } : { type: "set_paused", data: { paused: true } };
  if (action === "resume") return mode === "clip" ? { type: "set_autoplay", data: true } : { type: "set_paused", data: { paused: false } };
  if (action === "reset") return { type: "reset" };
  throw new Error(`no command ${action}`);
}
