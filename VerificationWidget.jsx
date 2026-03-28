/**
 * Smart Gov Biometric Verification Widget
 * ----------------------------------------
 * Real-time webcam → WebSocket → FastAPI pipeline.
 * Handles: auto pose detection, blink challenge, texture liveness, result display.
 *
 * Usage:
 *   <VerificationWidget nationalId="0602081696081" apiBase="http://localhost:8000" />
 */

import { useState, useEffect, useRef, useCallback } from "react";

const POSE_LABELS = { front: "Face Forward", left: "Turn Left", right: "Turn Right" };
const POSE_ICONS  = { front: "⬤", left: "◀", right: "▶" };

const DECISION_CONFIG = {
  "Express Approved":       { color: "#00C896", bg: "#002B22", label: "Verified" },
  "Manual Review":          { color: "#F5A623", bg: "#2B1F00", label: "Under Review" },
  "More Information Needed":{ color: "#E05C5C", bg: "#2B0808", label: "Not Verified" },
};

export default function VerificationWidget({ nationalId = "", apiBase = "http://localhost:8000" }) {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const wsRef       = useRef(null);
  const streamRef   = useRef(null);
  const blinkRef    = useRef({ prevEAR: null, belowCount: 0, confirmed: false });
  const rafRef      = useRef(null);

  const [phase, setPhase]             = useState("idle");        // idle | init | challenge | capturing | verifying | result | error
  const [instruction, setInstruction] = useState("");
  const [targetPose, setTargetPose]   = useState("front");
  const [posesDone, setPosesDone]     = useState(0);
  const [posesTotal, setPosesTotal]   = useState(3);
  const [blinkDone, setBlinkDone]     = useState(false);
  const [blinkCount, setBlinkCount]   = useState(0);
  const [stableProgress, setStableProgress] = useState(0);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState("");
  const [faceBox, setFaceBox]         = useState(null);
  const [poseOk, setPoseOk]           = useState(false);
  const [qualityOk, setQualityOk]     = useState(false);

  // ── Camera helpers ──
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      return true;
    } catch {
      setError("Camera access denied. Please allow camera permissions.");
      setPhase("error");
      return false;
    }
  };

  const stopCamera = () => {
    cancelAnimationFrame(rafRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  };

  // ── Blink detection (EAR approximation via face bounding box height ratio) ──
  // Real EAR needs landmark points; here we approximate by watching the
  // bounding box aspect ratio change — a blink compresses it momentarily.
  // In production, replace with MediaPipe FaceMesh for precise EAR.
  const processBlink = useCallback((bb) => {
    if (!bb || blinkRef.current.confirmed) return;
    const [, , w, h] = bb;
    const ear = w > 0 ? h / w : 0;
    const { prevEAR, belowCount } = blinkRef.current;

    const BLINK_THRESH = 1.05; // height suddenly drops (eyes close)
    if (prevEAR !== null && ear < prevEAR * BLINK_THRESH - 0.1) {
      blinkRef.current.belowCount = belowCount + 1;
    } else if (blinkRef.current.belowCount >= 1 && ear >= prevEAR * BLINK_THRESH - 0.1) {
      const newCount = blinkCount + 1;
      setBlinkCount(newCount);
      if (newCount >= 1) {
        blinkRef.current.confirmed = true;
        setBlinkDone(true);
      }
      blinkRef.current.belowCount = 0;
    }
    blinkRef.current.prevEAR = ear;
  }, [blinkCount]);

  // ── Frame capture & WS send ──
  const sendFrame = useCallback(() => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    const ws     = wsRef.current;

    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;

    const ctx = canvas.getContext("2d");
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);
    const frame = canvas.toDataURL("image/jpeg", 0.85);
    ws.send(JSON.stringify({ type: "analysis", frame }));
  }, []);

  // ── Draw overlay on video ──
  const drawOverlay = useCallback((bb, ok) => {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);

    if (bb) {
      const [x, y, w, h] = bb;
      ctx.strokeStyle = ok ? "#00C896" : "#F5A623";
      ctx.lineWidth   = 3;
      ctx.beginPath();
      // Corner brackets instead of full rect — cleaner
      const cs = 20;
      ctx.moveTo(x, y + cs);   ctx.lineTo(x, y);   ctx.lineTo(x + cs, y);
      ctx.moveTo(x+w-cs, y);   ctx.lineTo(x+w, y); ctx.lineTo(x+w, y+cs);
      ctx.moveTo(x+w, y+h-cs); ctx.lineTo(x+w, y+h); ctx.lineTo(x+w-cs, y+h);
      ctx.moveTo(x+cs, y+h);   ctx.lineTo(x, y+h); ctx.lineTo(x, y+h-cs);
      ctx.stroke();
    }
  }, []);

  // ── WebSocket ──
  const connectWS = useCallback(() => {
    const wsUrl = `${apiBase.replace(/^http/, "ws")}/ws/stream/${nationalId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setPhase("challenge");
      setInstruction("Blink once to prove you're live");

      // Send frames at ~10fps
      const loop = () => {
        sendFrame();
        rafRef.current = setTimeout(loop, 100);
      };
      rafRef.current = setTimeout(loop, 100);
    };

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);

      if (msg.type === "feedback") {
        setInstruction(msg.instruction || "");
        setPoseOk(msg.pose_ok);
        setQualityOk(msg.quality_ok);
        if (msg.bounding_box) {
          setFaceBox(msg.bounding_box);
          processBlink(msg.bounding_box);
          drawOverlay(msg.bounding_box, msg.pose_ok && msg.quality_ok);
        }
        if (typeof msg.stable_frames === "number") {
          setStableProgress(msg.stable_frames / (msg.stable_needed || 8));
        }
      }

      if (msg.type === "pose_captured") {
        setPosesDone(msg.poses_done);
        setTargetPose(msg.next_pose);
        setInstruction(msg.instruction);
        setStableProgress(0);
      }

      if (msg.type === "result") {
        clearTimeout(rafRef.current);
        setPhase("result");
        setResult(msg);
        stopCamera();
      }

      if (msg.type === "error") {
        setError(msg.message);
        setPhase("error");
        stopCamera();
      }
    };

    ws.onclose = () => {
      clearTimeout(rafRef.current);
    };

    ws.onerror = () => {
      setError("Connection to verification server lost.");
      setPhase("error");
    };
  }, [nationalId, apiBase, sendFrame, processBlink, drawOverlay]);

  const start = async () => {
    setError("");
    setResult(null);
    setPhase("init");
    setPosesDone(0);
    setTargetPose("front");
    setBlinkDone(false);
    setBlinkCount(0);
    setStableProgress(0);
    blinkRef.current = { prevEAR: null, belowCount: 0, confirmed: false };

    const ok = await startCamera();
    if (ok) connectWS();
  };

  const reset = () => {
    if (wsRef.current) wsRef.current.close();
    stopCamera();
    setPhase("idle");
    setResult(null);
    setError("");
    setFaceBox(null);
    setInstruction("");
    setStableProgress(0);
  };

  useEffect(() => () => { if (wsRef.current) wsRef.current.close(); stopCamera(); }, []);

  // ── Render ──
  const cfg = result ? DECISION_CONFIG[result.decision] || DECISION_CONFIG["Manual Review"] : null;

  return (
    <div style={styles.root}>
      <div style={styles.card}>
        {/* Header */}
        <div style={styles.header}>
          <div style={styles.logo}>
            <span style={styles.logoDot} />
            <span style={styles.logoText}>SmartGov</span>
          </div>
          <span style={styles.headerTag}>Biometric Verification</span>
        </div>

        {/* Camera area */}
        <div style={styles.cameraWrap}>
          <video
            ref={videoRef}
            style={{ ...styles.video, display: phase !== "idle" && phase !== "result" && phase !== "error" ? "block" : "none" }}
            muted
            playsInline
          />
          <canvas
            ref={canvasRef}
            style={{ ...styles.canvas, display: phase !== "idle" && phase !== "result" && phase !== "error" ? "block" : "none" }}
          />

          {/* Idle state */}
          {phase === "idle" && (
            <div style={styles.placeholder}>
              <div style={styles.faceIcon}>
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                  <circle cx="32" cy="32" r="28" stroke="#334" strokeWidth="1.5" strokeDasharray="4 3"/>
                  <circle cx="22" cy="26" r="4" fill="#445"/>
                  <circle cx="42" cy="26" r="4" fill="#445"/>
                  <path d="M20 42 Q32 52 44 42" stroke="#445" strokeWidth="2" strokeLinecap="round" fill="none"/>
                </svg>
              </div>
              <p style={styles.placeholderText}>Ready to verify identity</p>
            </div>
          )}

          {/* Result overlay */}
          {phase === "result" && cfg && (
            <div style={{ ...styles.resultOverlay, background: cfg.bg }}>
              <div style={{ ...styles.resultBadge, borderColor: cfg.color, color: cfg.color }}>
                {cfg.label}
              </div>
              <div style={styles.resultScore}>
                <span style={styles.scoreNum}>{result.score}</span>
                <span style={styles.scoreLabel}>/ 100</span>
              </div>
              <div style={styles.resultDecision}>{result.decision}</div>
              <div style={styles.resultBreakdown}>
                <MetricRow label="Biometric" value={result.biometric?.score} max={50} color={cfg.color}/>
                <MetricRow label="Liveness"  value={result.liveness?.score}  max={25} color={cfg.color}/>
                <MetricRow label="Device"    value={result.device?.score}     max={15} color={cfg.color}/>
                <MetricRow label="Anti-spoof" value={result.anti_spoofing?.texture_score} max={30} color={cfg.color}/>
              </div>
            </div>
          )}

          {phase === "error" && (
            <div style={styles.errorOverlay}>
              <span style={styles.errorIcon}>✕</span>
              <p style={styles.errorMsg}>{error}</p>
            </div>
          )}
        </div>

        {/* Status bar */}
        {(phase === "challenge" || phase === "capturing") && (
          <div style={styles.statusBar}>
            {/* Blink indicator */}
            <div style={styles.statusItem}>
              <div style={{ ...styles.statusDot, background: blinkDone ? "#00C896" : "#556" }} />
              <span style={styles.statusLabel}>{blinkDone ? "Blink ✓" : "Blink once"}</span>
            </div>

            {/* Pose steps */}
            <div style={styles.poseSteps}>
              {["front","left","right"].map((p, i) => (
                <div key={p} style={{
                  ...styles.poseStep,
                  background: i < posesDone ? "#00C896" : (p === targetPose ? "#334" : "transparent"),
                  borderColor: i < posesDone ? "#00C896" : (p === targetPose ? "#667" : "#334"),
                }}>
                  <span style={{ fontSize: 10, color: i < posesDone ? "#001" : "#aab" }}>
                    {POSE_ICONS[p]}
                  </span>
                </div>
              ))}
            </div>

            {/* Stable progress */}
            <div style={styles.progressWrap}>
              <div style={{ ...styles.progressBar, width: `${Math.round(stableProgress * 100)}%` }} />
            </div>
          </div>
        )}

        {/* Instruction */}
        {instruction && phase !== "result" && phase !== "idle" && (
          <div style={styles.instruction}>{instruction}</div>
        )}

        {/* Action button */}
        <div style={styles.actions}>
          {phase === "idle" && (
            <button style={styles.btn} onClick={start} disabled={!nationalId}>
              Begin Verification
            </button>
          )}
          {(phase === "result" || phase === "error") && (
            <button style={{ ...styles.btn, background: "#1a1a2a" }} onClick={reset}>
              Verify Again
            </button>
          )}
          {(phase === "challenge" || phase === "capturing" || phase === "init") && (
            <button style={{ ...styles.btn, background: "#1a1a2a", color: "#667" }} onClick={reset}>
              Cancel
            </button>
          )}
        </div>

        {/* ID display */}
        {nationalId && (
          <div style={styles.idTag}>
            ID: <span style={styles.idNum}>{nationalId}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricRow({ label, value, max, color }) {
  const pct = Math.min(((value || 0) / max) * 100, 100);
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 11, color: "#889" }}>{label}</span>
        <span style={{ fontSize: 11, color: "#ccd" }}>{value ?? "—"}</span>
      </div>
      <div style={{ height: 3, background: "#223", borderRadius: 2 }}>
        <div style={{ height: 3, width: `${pct}%`, background: color, borderRadius: 2, transition: "width .6s" }}/>
      </div>
    </div>
  );
}

const styles = {
  root: {
    minHeight: "100vh",
    background: "#0a0b14",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif",
    padding: "24px 16px",
  },
  card: {
    width: "100%",
    maxWidth: 480,
    background: "#111220",
    borderRadius: 20,
    overflow: "hidden",
    border: "1px solid #1e2035",
    boxShadow: "0 24px 64px rgba(0,0,0,.6)",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "16px 20px",
    borderBottom: "1px solid #1a1b2e",
  },
  logo: { display: "flex", alignItems: "center", gap: 8 },
  logoDot: {
    width: 8, height: 8,
    borderRadius: "50%",
    background: "#00C896",
    boxShadow: "0 0 8px #00C896",
  },
  logoText: { fontSize: 14, fontWeight: 600, color: "#dde", letterSpacing: "0.04em" },
  headerTag: { fontSize: 11, color: "#556", letterSpacing: "0.06em", textTransform: "uppercase" },

  cameraWrap: {
    position: "relative",
    width: "100%",
    aspectRatio: "4/3",
    background: "#0a0b10",
    overflow: "hidden",
  },
  video: {
    position: "absolute", inset: 0,
    width: "100%", height: "100%",
    objectFit: "cover",
    transform: "scaleX(-1)", // mirror
  },
  canvas: {
    position: "absolute", inset: 0,
    width: "100%", height: "100%",
    objectFit: "cover",
    transform: "scaleX(-1)",
    pointerEvents: "none",
  },

  placeholder: {
    position: "absolute", inset: 0,
    display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    gap: 16,
  },
  faceIcon: { opacity: 0.4 },
  placeholderText: { color: "#445", fontSize: 13, margin: 0 },

  resultOverlay: {
    position: "absolute", inset: 0,
    display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    padding: 24, gap: 8,
  },
  resultBadge: {
    border: "1.5px solid",
    borderRadius: 20, padding: "4px 16px",
    fontSize: 11, fontWeight: 600,
    letterSpacing: "0.1em", textTransform: "uppercase",
  },
  resultScore: { display: "flex", alignItems: "baseline", gap: 4, marginTop: 8 },
  scoreNum: { fontSize: 56, fontWeight: 700, color: "#dde", lineHeight: 1 },
  scoreLabel: { fontSize: 18, color: "#556" },
  resultDecision: { fontSize: 13, color: "#889", marginBottom: 12 },
  resultBreakdown: { width: "100%", maxWidth: 260 },

  errorOverlay: {
    position: "absolute", inset: 0,
    display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    background: "#1a0808", gap: 12,
  },
  errorIcon: { fontSize: 32, color: "#E05C5C" },
  errorMsg: { color: "#c88", fontSize: 13, textAlign: "center", margin: 0, maxWidth: 280 },

  statusBar: {
    padding: "10px 20px",
    borderBottom: "1px solid #1a1b2e",
    display: "flex", alignItems: "center", gap: 12,
  },
  statusItem: { display: "flex", alignItems: "center", gap: 6 },
  statusDot: { width: 8, height: 8, borderRadius: "50%", flexShrink: 0 },
  statusLabel: { fontSize: 11, color: "#778" },
  poseSteps: { display: "flex", gap: 6, marginLeft: "auto" },
  poseStep: {
    width: 24, height: 24, borderRadius: 6,
    border: "1px solid", display: "flex",
    alignItems: "center", justifyContent: "center",
    transition: "all .2s",
  },
  progressWrap: {
    flex: 1, height: 3,
    background: "#1a1b2e", borderRadius: 2,
    overflow: "hidden",
  },
  progressBar: {
    height: "100%", background: "#00C896",
    borderRadius: 2, transition: "width .1s",
  },

  instruction: {
    padding: "12px 20px",
    fontSize: 13, color: "#99a", textAlign: "center",
    borderBottom: "1px solid #1a1b2e",
    minHeight: 42, display: "flex", alignItems: "center", justifyContent: "center",
  },

  actions: { padding: "16px 20px" },
  btn: {
    width: "100%", padding: "13px",
    background: "#00C896", color: "#001a14",
    border: "none", borderRadius: 10,
    fontSize: 14, fontWeight: 600,
    cursor: "pointer", letterSpacing: "0.02em",
    transition: "opacity .15s",
  },

  idTag: {
    padding: "8px 20px 14px",
    fontSize: 11, color: "#334", textAlign: "center",
  },
  idNum: { color: "#445", letterSpacing: "0.08em" },
};
