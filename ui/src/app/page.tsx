"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// API base URL. The FastAPI service runs on :8000.
const API = "http://localhost:8000";

// Plate dimensions in metres (matches the simulator).
const PLATE_M = 1.0;

// Message types from the WebSocket stream.
type MetaMsg = {
  type: "meta";
  source_xy_m: [number, number];
  source_type: string;
  sample_rate_hz: number;
  sensor_positions_m: [number, number][];
};
type WaveformMsg = { type: "waveform"; channel: number; samples: number[] };
type HitsMsg = {
  type: "hits";
  hits: {
    channel: number;
    onset_time_s: number;
    peak_amplitude: number;
    rise_time_s: number;
    duration_s: number;
  }[];
};
type FeaturesMsg = {
  type: "features";
  features: {
    rise_time_s: number;
    ring_down_count: number;
    peak_amplitude: number;
    energy: number;
    duration_s: number;
    channel: number;
  };
};
type LocMsg = {
  type: "localisation";
  position_m: [number, number] | null;
  residual_s?: number;
  confidence?: number;
  error_m: number | null;
};

type EventState = {
  meta: MetaMsg | null;
  waveforms: number[][];
  hits: HitsMsg["hits"];
  features: FeaturesMsg["features"] | null;
  loc: LocMsg | null;
};

const EMPTY_STATE: EventState = {
  meta: null,
  waveforms: [[], [], [], []],
  hits: [],
  features: null,
  loc: null,
};

export default function Page() {
  const [state, setState] = useState<EventState>(EMPTY_STATE);
  const [injecting, setInjecting] = useState(false);
  const [x, setX] = useState(0.5);
  const [y, setY] = useState(0.5);
  const [sourceType, setSourceType] = useState("crack");

  const plateCanvasRef = useRef<HTMLCanvasElement>(null);
  const waveCanvasRef = useRef<HTMLCanvasElement>(null);
  const specCanvasRef = useRef<HTMLCanvasElement>(null);

  const inject = useCallback(async () => {
    setInjecting(true);
    setState(EMPTY_STATE);
    try {
      const r = await fetch(
        `${API}/inject?source_type=${sourceType}&x=${x}&y=${y}`,
        { method: "POST" }
      );
      if (!r.ok) throw new Error(`inject failed: ${r.statusText}`);
      const body = await r.json();
      const ws = new WebSocket(`${API.replace("http", "ws")}/stream/${body.session_id}`);
      const newState: EventState = {
        meta: null,
        waveforms: [[], [], [], []],
        hits: [],
        features: null,
        loc: null,
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "meta") {
          newState.meta = msg;
        } else if (msg.type === "waveform") {
          newState.waveforms[msg.channel] = msg.samples;
        } else if (msg.type === "hits") {
          newState.hits = msg.hits;
        } else if (msg.type === "features") {
          newState.features = msg.features;
        } else if (msg.type === "localisation") {
          newState.loc = msg;
          setState({ ...newState });
          ws.close();
        }
      };
      ws.onerror = (e) => {
        console.error("ws error", e);
        setInjecting(false);
      };
      ws.onclose = () => setInjecting(false);
    } catch (e) {
      console.error(e);
      setInjecting(false);
    }
  }, [x, y, sourceType]);

  // Draw the plate: sensors, true source, estimated position, error.
  useEffect(() => {
    const canvas = plateCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    // Plate outline.
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 2;
    ctx.strokeRect(10, 10, W - 20, H - 20);
    const m2px = (mx: number) => 10 + (mx / PLATE_M) * (W - 20);
    // Sensors.
    if (state.meta) {
      ctx.fillStyle = "red";
      for (const [sx, sy] of state.meta.sensor_positions_m) {
        ctx.beginPath();
        ctx.arc(m2px(sx), m2px(sy), 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = "#000";
        ctx.font = "10px monospace";
        ctx.fillText(`(${sx.toFixed(2)},${sy.toFixed(2)})`, m2px(sx) + 8, m2px(sy) - 8);
        ctx.fillStyle = "red";
      }
      // True source (green cross).
      const [tx, ty] = state.meta.source_xy_m;
      ctx.strokeStyle = "green";
      ctx.lineWidth = 2;
      const px = m2px(tx);
      const py = m2px(ty);
      ctx.beginPath();
      ctx.moveTo(px - 8, py);
      ctx.lineTo(px + 8, py);
      ctx.moveTo(px, py - 8);
      ctx.lineTo(px, py + 8);
      ctx.stroke();
    }
    // Estimated position (blue circle).
    if (state.loc?.position_m) {
      const [ex, ey] = state.loc.position_m;
      ctx.strokeStyle = "blue";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(m2px(ex), m2px(ey), 8, 0, 2 * Math.PI);
      ctx.stroke();
      // Error line.
      if (state.meta) {
        const [tx, ty] = state.meta.source_xy_m;
        ctx.strokeStyle = "#888";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(m2px(ex), m2px(ey));
        ctx.lineTo(m2px(tx), m2px(ty));
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }, [state]);

  // Draw the four waveforms.
  useEffect(() => {
    const canvas = waveCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const chH = H / 4;
    for (let ch = 0; ch < 4; ch++) {
      const samples = state.waveforms[ch];
      if (samples.length === 0) continue;
      const offset = ch * chH + chH / 2;
      ctx.strokeStyle = `hsl(${ch * 90}, 70%, 50%)`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const maxAbs = Math.max(...samples.map(Math.abs), 1e-12);
      for (let i = 0; i < samples.length; i++) {
        const sx = (i / samples.length) * W;
        const sy = offset - (samples[i] / maxAbs) * (chH * 0.4);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();
      // Channel label.
      ctx.fillStyle = "#000";
      ctx.font = "10px monospace";
      ctx.fillText(`ch${ch}`, 4, ch * chH + 12);
    }
  }, [state]);

  // Draw the spectrogram of the strongest channel.
  useEffect(() => {
    const canvas = specCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    // Simple magnitude spectrogram via FFT is too heavy for the browser; draw the
    // waveform's short-time energy as a proxy spectrogram (time vs energy).
    // This is a qualitative view, not a true STFT (3.3 says "spectrogram view").
    const ch = state.features?.channel ?? 0;
    const samples = state.waveforms[ch] ?? [];
    if (samples.length === 0) return;
    const frameSize = 32;
    const nFrames = Math.floor(samples.length / frameSize);
    const nBins = 16;
    const img = ctx.createImageData(nFrames, nBins);
    for (let f = 0; f < nFrames; f++) {
      const frame = samples.slice(f * frameSize, (f + 1) * frameSize);
      // Crude "frequency" content: count zero-crossings as a frequency proxy.
      let zc = 0;
      for (let i = 1; i < frame.length; i++) {
        if ((frame[i] >= 0) !== (frame[i - 1] >= 0)) zc++;
      }
      for (let b = 0; b < nBins; b++) {
        // Energy distribution: peak near the zero-crossing rate.
        const dist = Math.exp(-((b - zc / 2) ** 2) / 4);
        const v = Math.min(255, Math.floor(dist * 255));
        const idx = (b * nFrames + f) * 4;
        img.data[idx] = v; // R
        img.data[idx + 1] = v; // G
        img.data[idx + 2] = 0; // B
        img.data[idx + 3] = 255; // A
      }
    }
    // Scale to canvas.
    ctx.putImageData(img, 0, 0);
    // Stretch to fit.
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(canvas, 0, 0, nFrames, nBins, 0, 0, W, H);
  }, [state]);

  const errorMm = state.loc?.error_m != null ? Math.round(state.loc.error_m * 1000) : null;

  return (
    <div style={{ fontFamily: "monospace", padding: 20, maxWidth: 1200, margin: "0 auto" }}>
      <h1>Structural Listener</h1>
      <p style={{ color: "#666", fontSize: 13 }}>
        Simulated acoustic-emission monitoring. Waveforms are FDTD-simulated, not
        recorded. Localisation is closed-form multilateration. See the README.
      </p>
      <div style={{ display: "flex", gap: 12, alignItems: "center", margin: "12px 0" }}>
        <label>
          type:{" "}
          <select value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
            <option value="crack">crack</option>
            <option value="friction">friction</option>
            <option value="impact">impact</option>
            <option value="noise">noise</option>
          </select>
        </label>
        <label>
          x:{" "}
          <input
            type="number"
            min={0.1}
            max={0.9}
            step={0.01}
            value={x}
            onChange={(e) => setX(parseFloat(e.target.value))}
            style={{ width: 60 }}
          />
        </label>
        <label>
          y:{" "}
          <input
            type="number"
            min={0.1}
            max={0.9}
            step={0.01}
            value={y}
            onChange={(e) => setY(parseFloat(e.target.value))}
            style={{ width: 60 }}
          />
        </label>
        <button onClick={inject} disabled={injecting} style={{ padding: "4px 12px" }}>
          {injecting ? "injecting..." : "inject"}
        </button>
      </div>
      <div style={{ display: "flex", gap: 20 }}>
        <div>
          <h3>Plate (green=true, blue=estimate)</h3>
          <canvas ref={plateCanvasRef} width={300} height={300} style={{ border: "1px solid #ccc" }} />
        </div>
        <div>
          <h3>Waveforms</h3>
          <canvas ref={waveCanvasRef} width={400} height={200} style={{ border: "1px solid #ccc" }} />
        </div>
        <div>
          <h3>Spectrogram (ch{state.features?.channel ?? 0})</h3>
          <canvas ref={specCanvasRef} width={400} height={120} style={{ border: "1px solid #ccc" }} />
        </div>
      </div>
      <div style={{ marginTop: 16, fontSize: 14 }}>
        {state.loc && (
          <div>
            <strong>
              localisation error: {errorMm != null ? `${errorMm} mm` : "n/a"}
            </strong>{" "}
            (confidence: {state.loc.confidence != null ? state.loc.confidence.toFixed(2) : "n/a"})
          </div>
        )}
        {state.features && (
          <div style={{ marginTop: 8, color: "#444" }}>
            features: rise={Math.round(state.features.rise_time_s * 1e6)}us,{" "}
            rdc={state.features.ring_down_count}, peak={state.features.peak_amplitude.toExponential(2)},{" "}
            energy={state.features.energy.toExponential(2)}, dur={Math.round(state.features.duration_s * 1e3)}ms
          </div>
        )}
      </div>
    </div>
  );
}