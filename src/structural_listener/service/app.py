"""FastAPI service: inject an event, stream the result over WebSocket (Sprint 3.1).

Endpoints:
- ``GET /`` \u2014 health check.
- ``POST /inject`` \u2014 run a simulated event; returns the source coords and a session id.
- ``WS /stream/{session_id}`` \u2014 stream the event's waveforms, hits, verdict, and
  localisation result to the client as JSON messages.

The service is one page (ADR-007): no routing beyond these three endpoints. The
Sprint 3 UI subscribes to the WebSocket and renders the live event.

Per ARCHITECTURE.md \u00a71 the demo flow is: inject at a coordinate \u2192 four waveforms
arrive at different times \u2192 classify \u2192 localise \u2192 show the error in mm. This service
exposes that flow as JSON messages over a single WebSocket.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from structural_listener.cli.inject import inject_event
from structural_listener.dsp.hit_detection import detect_hits
from structural_listener.dsp.tdoa import TdoaConfig, pick_arrival_times
from structural_listener.localize.multilateration import solve
from structural_listener.model.dataset import SOURCE_TYPES
from structural_listener.model.features import extract_features

# Effective wave speed for the gated envelope-peak picker (docs/sources.md).
C_EFF_M_PER_S = 1200.0

app = FastAPI(title="Structural Listener", version="0.1.0")
# CORS for the Next.js dev server (localhost:3000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of completed injections, keyed by session id.
# A real service would use a message broker; this is a demo (ADR-007: one page).
_SESSIONS: dict[str, dict[str, Any]] = {}


@dataclass
class InjectResponse:
    """Response to POST /inject."""

    session_id: str
    source_xy_m: tuple[float, float]
    source_type: str
    sample_rate_hz: float
    sensor_positions_m: list[list[float]]


@app.get("/")
def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.post("/inject")
def inject(source_type: str = "crack", x: float = 0.5, y: float = 0.5) -> InjectResponse:
    """Run a simulated event and store the result for the WebSocket stream."""
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unknown source type {source_type!r}")
    duration_s = 10.0e-3 if source_type == "noise" else 4.0e-3
    out_dir = Path("/tmp/structural_listener_service")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = inject_event(source_type, x, y, duration_s=duration_s, out_dir=out_dir)
    with np.load(path) as data:
        wf = np.asarray(data["waveforms"])
        sr = float(data["sample_rate_hz"])
        src = (float(data["source_xy_m"][0]), float(data["source_xy_m"][1]))
        sensors = np.asarray(data["sensor_positions_m"]).tolist()

    # Pre-compute the full pipeline result so the WebSocket just streams it.
    hits = detect_hits(wf, sr)
    arrival_times = pick_arrival_times(wf, sr, TdoaConfig())
    feats = extract_features(wf, sr)
    try:
        loc_result = solve(np.asarray(sensors), arrival_times, C_EFF_M_PER_S)
        error_m = float(
            np.hypot(loc_result.position_m[0] - src[0], loc_result.position_m[1] - src[1])
        )
        loc_payload = {
            "position_m": list(loc_result.position_m),
            "residual_s": loc_result.residual,
            "confidence": loc_result.confidence,
            "error_m": error_m,
        }
    except ValueError:
        loc_payload = {"position_m": None, "error_m": None}

    feats_payload = asdict(feats) if feats is not None else None

    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "source_xy_m": src,
        "source_type": source_type,
        "sample_rate_hz": sr,
        "sensor_positions_m": sensors,
        "waveforms": wf.tolist(),
        "hits": [
            {
                "channel": h.channel,
                "onset_time_s": h.onset_time_s,
                "peak_amplitude": h.peak_amplitude,
                "rise_time_s": h.rise_time_s,
                "duration_s": h.duration_s,
            }
            for h in hits
        ],
        "features": feats_payload,
        "localisation": loc_payload,
    }
    return InjectResponse(
        session_id=session_id,
        source_xy_m=src,
        source_type=source_type,
        sample_rate_hz=sr,
        sensor_positions_m=sensors,
    )


@app.websocket("/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: str) -> None:
    """Stream the event's waveforms, hits, verdict, and localisation as JSON messages.

    Message types (one JSON object per line):
    - ``{"type": "meta", ...}`` \u2014 source coords, sample rate, sensor positions.
    - ``{"type": "waveform", "channel": ch, "samples": [...]}`` \u2014 one per channel.
    - ``{"type": "hits", "hits": [...]}`` \u2014 the detected hits.
    - ``{"type": "features", "features": {...}}` \u2014 the classical AE features.
    - ``{"type": "localisation", ...}`` \u2014 the multilateration result + error.
    """
    await websocket.accept()
    session = _SESSIONS.get(session_id)
    if session is None:
        await websocket.send_text(json.dumps({"type": "error", "message": "unknown session"}))
        await websocket.close()
        return
    try:
        # Meta.
        await websocket.send_text(
            json.dumps(
                {
                    "type": "meta",
                    "source_xy_m": list(session["source_xy_m"]),
                    "source_type": session["source_type"],
                    "sample_rate_hz": session["sample_rate_hz"],
                    "sensor_positions_m": session["sensor_positions_m"],
                }
            )
        )
        # Waveforms: send each channel separately (downsample for the UI).
        # The UI doesn't need 824k samples; decimate by 100 to ~8 kHz for display.
        dec = 100
        for ch, samples in enumerate(session["waveforms"]):
            decimated = samples[::dec]
            await websocket.send_text(
                json.dumps({"type": "waveform", "channel": ch, "samples": decimated})
            )
        # Hits.
        await websocket.send_text(json.dumps({"type": "hits", "hits": session["hits"]}))
        # Features.
        if session["features"] is not None:
            await websocket.send_text(
                json.dumps({"type": "features", "features": session["features"]})
            )
        # Localisation.
        await websocket.send_text(json.dumps({"type": "localisation", **session["localisation"]}))
    except WebSocketDisconnect:
        return
