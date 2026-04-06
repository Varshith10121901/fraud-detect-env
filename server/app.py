"""
server/app.py — OpenEnv Environment Server

This file is ONLY an environment server:
  - NO LLM calls
  - NO inference logic
  - Serves the dashboard via an external HTML file (server/templates/index.html)

Endpoints:
  POST /reset   → start new episode, return {"observation": {...}}
  POST /step    → score agent response, return reward + done
  GET  /state   → current episode state
  GET  /system  → environment variable checks
  GET  /health  → {"status": "ok"}
  GET  /        → dashboard HTML (read-only, from templates/index.html)
"""

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── Data ────────────────────────────────────────────────────────────────

DATA_FILE      = Path(__file__).parent / "data" / "transactions.json"

def _load_transactions() -> List[Dict[str, Any]]:
    with open(DATA_FILE, "r") as f:
        return json.load(f)

TRANSACTIONS: List[Dict[str, Any]] = _load_transactions()

GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "TXN-001": {"label": "FRAUD", "fraud_type": "CARD_NOT_PRESENT",  "risk": "HIGH"},
    "TXN-002": {"label": "LEGIT", "fraud_type": None,                "risk": "LOW"},
    "TXN-003": {"label": "FRAUD", "fraud_type": "MONEY_LAUNDERING",  "risk": "HIGH"},
    "TXN-004": {"label": "LEGIT", "fraud_type": None,                "risk": "LOW"},
    "TXN-005": {"label": "FRAUD", "fraud_type": "ACCOUNT_TAKEOVER",  "risk": "HIGH"},
    "TXN-006": {"label": "LEGIT", "fraud_type": None,                "risk": "LOW"},
    "TXN-007": {"label": "FRAUD", "fraud_type": "PHISHING",          "risk": "HIGH"},
    "TXN-008": {"label": "FRAUD", "fraud_type": "IDENTITY_THEFT",    "risk": "HIGH"},
    "TXN-009": {"label": "LEGIT", "fraud_type": None,                "risk": "LOW"},
    "TXN-010": {"label": "FRAUD", "fraud_type": "CARD_NOT_PRESENT",  "risk": "HIGH"},
}

VALID_LABELS      = {"FRAUD", "LEGIT"}
VALID_FRAUD_TYPES = {"CARD_NOT_PRESENT", "ACCOUNT_TAKEOVER", "MONEY_LAUNDERING",
                     "IDENTITY_THEFT", "PHISHING"}

TASK_SEQUENCE = ["classify", "identify_type", "action_plan"]

# ── Episode State ────────────────────────────────────────────────────────

class EpisodeState:
    def __init__(self) -> None:
        self.transaction: Optional[Dict[str, Any]] = None
        self.task_index: int = 0
        self.rewards: List[float] = []
        self.done: bool = False
        self.steps: int = 0
        self.global_index: int = 0
        self.all_processed: bool = False
        self.predicted_label: Optional[str] = None
        self.history: List[Dict[str, Any]] = []

    def reset(self) -> Dict[str, Any]:
        if self.global_index >= len(TRANSACTIONS):
            self.all_processed = True
            return {"all_processed": True, "done": True, "current_task": "done"}

        self.transaction = TRANSACTIONS[self.global_index]
        self.global_index += 1
        self.task_index  = 0
        self.rewards     = []
        self.done        = False
        self.steps       = 0
        self.predicted_label = None
        return self._observation()

    def _observation(self) -> Dict[str, Any]:
        if getattr(self, "all_processed", False):
            return {"all_processed": True, "done": True, "current_task": "done"}
        if self.transaction is None:
            return {"error": "No episode active. Call /reset first."}
        return {
            "transaction":  self.transaction,
            "current_task": TASK_SEQUENCE[self.task_index] if not self.done else "done",
            "task_index":   self.task_index,
            "steps_taken":  self.steps,
            "done":         self.done,
        }

    def step(self, task: str, response: str) -> Dict[str, Any]:
        if self.done:
            return {"observation": self._observation(), "reward": 0.0,
                    "done": True, "info": {"error": "Episode already done"}}

        if self.transaction is None:
            raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

        expected_task = TASK_SEQUENCE[self.task_index]
        if task != expected_task:
            return {
                "observation": self._observation(),
                "reward": 0.0,
                "done": False,
                "info": {"error": f"Wrong task. Expected '{expected_task}', got '{task}'"},
            }

        txn_id  = self.transaction["transaction_id"]
        truth   = GROUND_TRUTH.get(txn_id, {})
        reward  = self._score(task, response.strip().upper(), truth)

        self.rewards.append(reward)
        self.steps += 1
        self.task_index += 1

        if task == "classify":
            self.predicted_label = response.strip().upper()
            if self.predicted_label == "LEGIT":
                self.done = True
        elif self.task_index >= len(TASK_SEQUENCE):
            self.done = True
            
        if self.done:
            self.history.append({
                "transaction_id": txn_id,
                "amount": self.transaction.get("amount"),
                "predicted_label": self.predicted_label,
                "truth_label": truth.get("label"),
                "score": sum(self.rewards) / len(self.rewards) if self.rewards else 0.0
            })

        return {
            "observation": self._observation(),
            "reward":      reward,
            "done":        self.done,
            "info":        {"task": task, "scored_response": response},
        }

    @staticmethod
    def _score(task: str, response: str, truth: Dict[str, Any]) -> float:
        if task == "classify":
            return 1.0 if response == truth.get("label") else 0.0
        if task == "identify_type":
            return 1.0 if response == truth.get("fraud_type") else 0.0
        if task == "action_plan":
            required_keywords = ["RISK LEVEL", "RECOMMENDED ACTION", "NEXT STEPS", "DO NOT"]
            text_upper = response.upper()
            matches = sum(1 for kw in required_keywords if kw in text_upper)
            return round(0.2 * matches, 2)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_active": self.transaction is not None,
            "transaction_id": self.transaction["transaction_id"] if self.transaction else None,
            "current_task":   TASK_SEQUENCE[self.task_index] if not self.done else "done",
            "task_index":     self.task_index,
            "steps":          self.steps,
            "rewards":        self.rewards,
            "done":           self.done,
            "all_processed":  getattr(self, "all_processed", False),
            "predicted_label": self.predicted_label
        }


STATE = EpisodeState()

# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection",
    description="OpenEnv environment server. No LLM calls here.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class StepRequest(BaseModel):
    task: str
    response: str

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/system")
async def system_status():
    """Return env-var status for the dashboard requirements panel."""
    return {
        "model_name": os.getenv("MODEL_NAME", "Unknown Model"),
        "has_token":  bool(os.getenv("HF_TOKEN")),
        "env_loaded": bool(os.getenv("API_BASE_URL")),
    }


@app.post("/reset")
async def reset():
    """Pick a random transaction and start a new episode."""
    obs = STATE.reset()
    return {"observation": obs}    # must be wrapped — OpenEnv validator requires this


@app.post("/step")
async def step(body: StepRequest):
    """Score the agent's response for the current task."""
    return STATE.step(task=body.task, response=body.response)


@app.get("/state")
async def get_state():
    """Return the full current episode state (read-only)."""
    return STATE.to_dict()


@app.get("/results")
async def get_results():
    """Return all processed transaction results."""
    return {"history": STATE.history, "total": len(TRANSACTIONS)}


@app.get("/rewards/summary")
async def rewards_summary():
    """Return aggregate statistics from all saved episodes."""
    from fraud_detect_env.rewards_logger import get_summary
    return get_summary()


@app.get("/rewards/episodes")
async def rewards_episodes(limit: int = 10):
    """Return recent episode results."""
    from fraud_detect_env.rewards_logger import list_episodes
    return {"episodes": list_episodes(limit=limit)}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the embedded dashboard HTML with no-cache headers."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MR.INSPECTOR | Real-Time Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
/* ── Reset & Enlarged Scale (103%) ───────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html { font-size: 103%; -webkit-font-smoothing: antialiased; }

:root{
  /* Dark Green & Black Palette */
  --bg: #010402;
  --card: rgba(4, 15, 8, 0.7);
  --card-hover: rgba(4, 15, 8, 0.9);
  
  /* Attractive, Vibrant Green Colors */
  --accent: #10b981;
  --accent-glow: rgba(16, 185, 129, 0.3);
  --accent-light: #34d399;
  --success: #059669;
  --danger: #ef4444; /* Keep red for fraud alerts */
  --warning: #f59e0b;
  
  --text-main: #f0fdf4;
  --text-muted: #6ee7b7;
  --text-faint: #047857;
  
  --border: rgba(16, 185, 129, 0.3);
  --border-strong: rgba(16, 185, 129, 0.6);
  
  --radius-lg: 20px;
  --radius-md: 12px;
  --radius-sm: 8px;
  
  --shadow-sm: 0 4px 6px -1px rgba(0,0,0,0.5);
  --shadow-md: 0 10px 25px -5px rgba(0,0,0,0.8);
  
  --font-sans: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

body {
  font-family: var(--font-sans);
  background-color: var(--bg);
  color: var(--text-main);
  min-height: 100dvh;
  display: flex;
  flex-direction: column;
  overflow-x: hidden;
}

/* Premium Texture & Background */
body::before {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: -2;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.03'/%3E%3C/svg%3E");
}
body::after {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: -1;
  background:
    radial-gradient(circle at 15% 50%, rgba(16,185,129,0.08), transparent 40%),
    radial-gradient(circle at 85% 30%, rgba(16,185,129,0.05), transparent 40%);
}
/* ── Custom Buttons and Modals ───────────────────────────────────── */
.btn-primary {
  background: var(--accent); color: white; border: none; padding: 10px 20px;
  border-radius: var(--radius-sm); font-size: 1rem; font-weight: 700;
  cursor: pointer; transition: all 0.2s ease;
  box-shadow: 0 4px 15px var(--accent-glow);
}
.btn-primary:hover { background: var(--accent-hover); transform: translateY(-2px); }

.modal {
  position: fixed; inset: 0; background: rgba(2,6,23,0.8); z-index: 1000;
  display: flex; justify-content: center; align-items: center; backdrop-filter: blur(5px);
}
.modal-content {
  background: var(--card); padding: 2rem; border-radius: var(--radius-lg);
  border: 1px solid var(--border-strong); max-width: 900px; width: 90%;
  max-height: 80vh; overflow-y: auto; color: var(--text-main); position: relative;
}
.close-btn { position: absolute; right: 20px; top: 15px; font-size: 2rem; cursor: pointer; color: var(--text-muted); }
.close-btn:hover { color: var(--danger); }

table.results-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
table.results-table th, table.results-table td { padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }
table.results-table th { color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; }
table.results-table tr:hover { background: rgba(255,255,255,0.02); }
.label-fraud { color: var(--danger); font-weight: bold; }
.label-legit { color: var(--success); font-weight: bold; }
/* ────────────────────────────────────────────────────────────────── */
/* ── Centered Cinematic Intro ───────────────────────────────────── */
#intro-overlay {
  position:fixed; inset:0; z-index:99999;
  background-color: #010402;
  display:flex; flex-direction:column; justify-content:center; align-items:center;
  color:#fff;
  transition: opacity 1s ease, visibility 1s ease;
}
.intro-text {
  font-size: clamp(2rem, 5vw, 3.5rem); 
  font-weight:900; letter-spacing:0.5em;
  text-transform:uppercase; white-space:nowrap;
  background: linear-gradient(to right, #ffffff, #6ee7b7);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: cinematicText 3.5s ease forwards;
}
.intro-sub {
  font-size:0.85rem; font-weight:500; letter-spacing:0.3em;
  color:#059669; margin-top:1rem; opacity:0;
  animation: fadeInOut 3s ease 0.5s forwards;
}
@keyframes cinematicText {
  0% { opacity:0; transform:scale(0.85); filter:blur(12px); letter-spacing:0.1em; }
  25% { opacity:1; transform:scale(1); filter:blur(0); letter-spacing:0.4em; }
  80% { opacity:1; transform:scale(1.05); filter:blur(0); letter-spacing:0.45em; }
  100% { opacity:0; transform:scale(1.2); filter:blur(8px); letter-spacing:0.7em; }
}
@keyframes fadeInOut {
  0% { opacity:0; } 20% { opacity:1; } 80% { opacity:1; } 100% { opacity:0; }
}

/* ── Full Screen Verdict Flash ──────────────────────────────────── */
#flash-overlay { position:fixed; inset:0; z-index:9990; pointer-events:none; opacity:0; }
.flash-red { animation: flashScreen 1s cubic-bezier(0.1, 0.8, 0.3, 1) forwards; background: rgba(239,68,68,0.25); box-shadow: inset 0 0 200px rgba(239,68,68,0.5); }
.flash-green { animation: flashScreen 1s cubic-bezier(0.1, 0.8, 0.3, 1) forwards; background: rgba(16,185,129,0.25); box-shadow: inset 0 0 200px rgba(16,185,129,0.4); }
@keyframes flashScreen { 0%{opacity:1; backdrop-filter:blur(4px);} 100%{opacity:0; backdrop-filter:blur(0px);} }

/* ── Header ─────────────────────────────────────────────────────── */
.hdr {
  background: rgba(1, 4, 2, 0.8); backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
  padding: 0 2rem; position: sticky; top: 0; z-index: 50;
}
.hdr-in {
  width: 100%; margin: 0 auto; height: 76px;
  display: flex; align-items: center; justify-content: space-between;
}
.logo {
  font-weight: 900; font-size: 1.35rem; letter-spacing: 0.05em;
  display: flex; align-items: center; gap: 14px; color: var(--text-main);
}
.logo-ic {
  width: 44px; height: 44px; border-radius: 12px;
  background: linear-gradient(135deg, #064e3b, #022c22);
  color: var(--accent-light); display: grid; place-items: center;
  border: 1px solid var(--border-strong);
  box-shadow: 0 0 15px rgba(16,185,129,0.2);
}
.status-badge {
  padding: 8px 16px; border-radius: 30px; font-size: 0.75rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.05em; display: flex; align-items: center; gap: 8px;
}
.status-badge.live { background: rgba(16,185,129,0.1); color: var(--accent-light); border: 1px solid rgba(16,185,129,0.25); }
.status-badge.offline { background: rgba(239,68,68,0.1); color: var(--danger); border: 1px solid rgba(239,68,68,0.25); }
.status-badge svg { width: 14px; height: 14px; }

/* ── Dashboard Full-Screen Layout ────────────────────────────────── */
.wrap { width: 100%; margin: 0 auto; padding: 2.5rem; flex: 1; display: flex; flex-direction: column; }
.dash-header { margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: flex-end; }
.dash-title h1 { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.4rem; letter-spacing: 0.02em; }
.dash-title p { color: var(--text-muted); font-size: 1rem; }

/* Grid: 2fr (Tx) | 1fr (State) | 1fr (Diag & Verdict) */
.grid-layout {
  display: grid; grid-template-columns: 2.2fr 1fr 1fr;
  grid-template-rows: auto auto; gap: 1.5rem; flex: 1;
}
.card-transaction { grid-column: 1 / 2; grid-row: 1 / 3; }
.card-episode     { grid-column: 2 / 3; grid-row: 1 / 2; }
.card-log         { grid-column: 2 / 3; grid-row: 2 / 3; align-self: start; }
.card-diagnostics { grid-column: 3 / 4; grid-row: 1 / 2; }
.card-verdict     { grid-column: 3 / 4; grid-row: 2 / 3; }

@media(max-width:1200px){ .grid-layout{grid-template-columns: 1.5fr 1fr;} .card-diagnostics, .card-verdict{grid-column: 2 / 3;} .card-log{grid-column: 1 / 3;} }
@media(max-width:850px){ .grid-layout{grid-template-columns: 1fr;} .card{grid-column: 1 / 2 !important; grid-row: auto !important;} }

/* ── Card Styles ─────────────────────────────────────────────────── */
.card {
  background: var(--card); backdrop-filter: blur(12px);
  border: 1px solid var(--border); border-radius: var(--radius-lg);
  padding: 2rem; box-shadow: var(--shadow-md);
  position: relative; overflow: hidden; transition: all 0.3s;
}
.card:hover { border-color: var(--border-strong); background: var(--card-hover); }
.card h3 {
  font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--text-muted); margin-bottom: 1.5rem; display: flex; align-items: center; gap: 10px;
}
.card h3 svg { color: var(--accent); width: 16px; height: 16px; }

/* ── Transaction Intel (Enlarged) ────────────────────────────────── */
.tx-id-row {
  background: rgba(0,0,0,0.4); border: 1px solid var(--border-strong);
  border-radius: var(--radius-md); padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
  display: flex; align-items: center; justify-content: space-between;
}
.tx-id-label { font-size: 0.85rem; font-weight: 800; color: var(--accent-light); letter-spacing: 0.1em; text-transform: uppercase; display: flex; align-items: center; gap: 8px; }
.tx-id-val { font-family: var(--font-mono); font-size: 1.5rem; font-weight: 700; color: #fff; }

.tx-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }
@media(max-width: 600px) { .tx-grid { grid-template-columns: 1fr 1fr; } }

.tx-field {
  background: rgba(255,255,255,0.02); border: 1px solid var(--border);
  padding: 1rem 1.25rem; border-radius: var(--radius-sm); transition: all 0.3s;
}
.tx-field.highlight { border-color: var(--accent); background: rgba(16,185,129,0.08); }
.tx-field.highlight-danger { border-color: var(--danger); background: rgba(239,68,68,0.05); }
.tx-label { font-size: 0.75rem; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; font-weight: 700; }
.tx-val { font-size: 1.15rem; color: #fff; font-family: var(--font-mono); font-weight: 600; }
.tx-val.muted { color: var(--text-muted); font-family: var(--font-sans); }

/* ── Detecting Radar Animations ──────────────────────────────────── */
.scanner-wrap {
  display: none; align-items: center; gap: 1.5rem;
  background: rgba(0,0,0,0.4); border: 1px solid var(--border);
  padding: 1.25rem; border-radius: var(--radius-md); margin-bottom: 1.5rem;
}
.scanner-wrap.active { display: flex; animation: slideDown 0.4s ease; }
@keyframes slideDown { from{opacity:0; transform:translateY(-10px);} to{opacity:1; transform:translateY(0);} }

.radar {
  width: 90px; height: 90px; border-radius: 50%; position: relative;
  border: 1px solid var(--accent); box-shadow: 0 0 20px var(--accent-glow);
  background: radial-gradient(circle, rgba(16,185,129,0.1) 0%, transparent 60%);
}
.radar::after {
  content: ''; position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(from 0deg, transparent 70%, rgba(16,185,129,0.6));
  animation: radarSpin 1.5s linear infinite;
}
@keyframes radarSpin { to { transform: rotate(360deg); } }

.scanner-info { flex: 1; }
.scan-title { font-size: 0.8rem; font-weight: 800; color: var(--accent-light); letter-spacing: 0.15em; margin-bottom: 8px; animation: pulseTxt 1.5s infinite; }
@keyframes pulseTxt { 0%,100%{opacity:0.6;} 50%{opacity:1;} }
.scan-line { height: 3px; background: rgba(16,185,129,0.15); border-radius: 2px; margin-bottom: 6px; overflow: hidden; position: relative; }
.scan-line::after {
  content: ''; position: absolute; top: 0; left: 0; height: 100%; width: 40%;
  background: var(--accent); animation: scanLine 2s ease-in-out infinite;
}
.scan-line:nth-child(3)::after { animation-delay: 0.4s; width: 70%; }
@keyframes scanLine { 0%{left: -50%;} 100%{left: 150%;} }

/* ── Progress Steps ──────────────────────────────────────────────── */
.progress-steps { display: flex; gap: 10px; margin-top: auto; }
.step { flex: 1; display: flex; flex-direction: column; gap: 8px; }
.step-bar { height: 4px; background: var(--border); border-radius: 2px; transition: 0.4s; }
.step-bar.active { background: var(--accent); box-shadow: 0 0 10px var(--accent-glow); }
.step-bar.done { background: var(--accent); opacity: 0.4; }
.step-label { font-size: 0.7rem; font-weight: 800; color: var(--text-faint); letter-spacing: 0.1em; text-transform: uppercase; }
.step-label.active { color: var(--accent-light); }

/* ── Verdict Display ─────────────────────────────────────────────── */
.verdict-display {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center; padding: 2.5rem 1.5rem; background: rgba(0,0,0,0.3);
  border: 1px solid var(--border); border-radius: var(--radius-md); transition: all 0.5s;
}
.verdict-display.fraud { background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.4); box-shadow: inset 0 0 30px rgba(239,68,68,0.15); }
.verdict-display.legit { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.4); box-shadow: inset 0 0 30px rgba(16,185,129,0.15); }

.verdict-icon {
  width: 64px; height: 64px; border-radius: 50%; display: grid; place-items: center;
  background: rgba(16,185,129,0.05); color: var(--text-faint); margin-bottom: 1rem;
  border: 1px solid var(--border); transition: all 0.4s;
}
.verdict-display.fraud .verdict-icon { color: var(--danger); background: rgba(239,68,68,0.2); border-color: var(--danger); animation: pop 0.5s cubic-bezier(0.2,0.8,0.2,1); }
.verdict-display.legit .verdict-icon { color: var(--success); background: rgba(16,185,129,0.2); border-color: var(--success); animation: pop 0.5s cubic-bezier(0.2,0.8,0.2,1); }
.verdict-display .verdict-icon svg { width: 32px; height: 32px; }
@keyframes pop { 0%{transform:scale(0) rotate(-20deg);} 100%{transform:scale(1) rotate(0);} }

.verdict-text { font-size: 1.4rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); }
.verdict-display.fraud .verdict-text { color: var(--danger); }
.verdict-display.legit .verdict-text { color: var(--success); }
.verdict-sub { font-size: 0.85rem; color: var(--text-faint); margin-top: 8px; font-family: var(--font-mono); }

/* ── Key-Value Rows ──────────────────────────────────────────────── */
.kv { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(16,185,129,0.1); font-size: 0.95rem; }
.kv:last-child { border: none; padding-bottom: 0; }
.kv .k { color: var(--text-muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
.kv .v { font-weight: 700; font-family: var(--font-mono); color: #fff; }

/* ── Dynamic Terminal Log with Toggle Button ─────────────────────── */
.log-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.btn-toggle {
  background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.4);
  color: var(--accent-light); padding: 6px 14px; border-radius: var(--radius-sm);
  font-size: 0.75rem; font-weight: 700; text-transform: uppercase; cursor: pointer;
  display: flex; align-items: center; gap: 8px; transition: 0.3s;
}
.btn-toggle:hover { background: rgba(16,185,129,0.25); color: #fff; }
.term-badge { width: 8px; height: 8px; border-radius: 50%; background: var(--danger); display: none; }
.term-badge.active { display: block; animation: pulseBadge 1s infinite; }
@keyframes pulseBadge { 0%,100%{opacity:1; transform:scale(1);} 50%{opacity:0.5; transform:scale(1.3);} }

.console-wrapper {
  max-height: 0; overflow: hidden; transition: max-height 0.4s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.console-wrapper.open { max-height: 350px; }

.console {
  background: #000201; border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 1rem; height: 250px;
  overflow-y: auto; font-family: var(--font-mono); font-size: 0.8rem;
  color: var(--text-muted); box-shadow: inset 0 0 20px rgba(0,0,0,0.7);
}
.console::-webkit-scrollbar { width: 4px; }
.console::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 2px; }
.console p { margin-bottom: 8px; line-height: 1.4; display: flex; gap: 8px; border-bottom: 1px solid rgba(16,185,129,0.1); padding-bottom: 8px; }
.console .ts { color: var(--text-faint); flex-shrink: 0; }
.console .msg { flex: 1; }
.console .info { color: #fff; }
.console .warn { color: var(--warning); }
.console .err  { color: var(--danger); }
.console .ok   { color: var(--accent-light); }

/* ── Toasts ──────────────────────────────────────────────────────── */
#toast-container { position: fixed; bottom: 2rem; right: 2rem; z-index: 9998; display: flex; flex-direction: column; gap: 1rem; }
.toast { background: var(--card); border: 1px solid var(--border); color: #fff; border-left: 4px solid var(--danger); padding: 1rem 1.5rem; border-radius: var(--radius-sm); box-shadow: var(--shadow-md); font-size: 0.9rem; font-weight: 600; animation: toastIn 0.3s cubic-bezier(0.175,0.885,0.32,1.275) forwards; }
@keyframes toastIn { from{transform:translateX(120%); opacity:0;} to{transform:translateX(0); opacity:1;} }
</style>
</head>
<body>

<!-- ── SVG Icon Dictionary (Hidden) ─────────────────────────────── -->
<svg style="display:none;" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <g id="ic-shield"><path d="M12 2L3 6v6c0 5.5 3.8 10.7 9 12 5.2-1.3 9-6.5 9-12V6l-9-4z" fill="none" stroke="currentColor" stroke-width="2"/><path d="M9 12l2 2 4-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></g>
    <g id="ic-target"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></g>
    <g id="ic-scan"><path d="M4 8V5h3M20 8V5h-3M4 16v3h3M20 16v3h-3" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="currentColor"/></g>
    <g id="ic-term"><polyline points="4 17 10 11 4 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><line x1="12" y1="19" x2="20" y2="19" stroke="currentColor" stroke-width="2"/></g>
    <g id="ic-check"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 12.5l2.5 2.5 5-5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></g>
    <g id="ic-alert"><path d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></g>
  </defs>
</svg>

<!-- ── Cinematic Intro (Centered) ───────────────────────────────── -->
<div id="intro-overlay">
  <div class="intro-text">MR.INSPECTOR</div>
  <div class="intro-sub">INITIALIZING SURVEILLANCE</div>
</div>

<!-- ── Full Screen Blink Overlay ────────────────────────────────── -->
<div id="flash-overlay"></div>

<!-- ── Header ───────────────────────────────────────────────────── -->
<header class="hdr">
  <div class="hdr-in">
    <div class="logo">
      <div class="logo-ic"><svg viewBox="0 0 24 24" width="22" height="22"><use href="#ic-shield"/></svg></div>
      MR.INSPECTOR
    </div>
    <div class="hdr-right">
      <div class="status-badge" id="net-status">
        <svg viewBox="0 0 24 24"><use href="#ic-check"/></svg> <span id="net-text">Checking...</span>
      </div>
    </div>
  </div>
</header>

<!-- ── Main Grid ─────────────────────────────────────────────────── -->
<div class="wrap">
  <div class="dash-header">
    <div class="dash-title">
      <h1>Surveillance Feed</h1>
      <p>Real-time neural inference & fraud tracking.</p>
    </div>
    <button id="btn-results" class="btn-primary" style="display:none;" onclick="showResults()">See Results</button>
  </div>

  <!-- Modal for Results -->
  <div id="results-modal" class="modal" style="display:none;">
    <div class="modal-content">
      <span class="close-btn" onclick="closeResults()">&times;</span>
      <h2>Final Analysis Results</h2>
      <div id="results-table-container"></div>
    </div>
  </div>

  <div class="grid-layout">
    
    <!-- ── Transaction Intel Card (Large) ───────────────────────── -->
    <div class="card card-transaction">
      <h3><svg viewBox="0 0 24 24"><use href="#ic-target"/></svg> Target Intelligence</h3>
      
      <div class="tx-id-row" id="tx-id-row">
        <div class="tx-id-label"><svg viewBox="0 0 24 24" width="18" height="18"><use href="#ic-target"/></svg> TRACE ID</div>
        <div class="tx-id-val" id="tx-id-val">AWAITING_DATA</div>
      </div>

      <!-- Radar Animation -->
      <div class="scanner-wrap" id="scanner">
        <div class="radar"></div>
        <div class="scanner-info">
          <div class="scan-title">TRACKING ANOMALY...</div>
          <div class="scan-line"><div class="bar"></div></div>
          <div class="scan-line"><div class="bar"></div></div>
        </div>
      </div>

      <!-- Data Grid -->
      <div class="tx-grid">
        <div class="tx-field"><div class="tx-label">Amount</div><div class="tx-val muted" id="tx-amount">—</div></div>
        <div class="tx-field"><div class="tx-label">Merchant</div><div class="tx-val muted" id="tx-merchant">—</div></div>
        <div class="tx-field"><div class="tx-label">Category</div><div class="tx-val muted" id="tx-category">—</div></div>
        <div class="tx-field"><div class="tx-label">Location</div><div class="tx-val muted" id="tx-location">—</div></div>
        <div class="tx-field"><div class="tx-label">Card Type</div><div class="tx-val muted" id="tx-card">—</div></div>
        <div class="tx-field"><div class="tx-label">Time</div><div class="tx-val muted" id="tx-time">—</div></div>
        <div class="tx-field" id="wrap-steps"><div class="tx-label">Steps Done</div><div class="tx-val" id="tx-steps" style="color:var(--accent-light)">0</div></div>
        <div class="tx-field" id="wrap-ep"><div class="tx-label">Engine</div><div class="tx-val" id="tx-episode" style="color:var(--text-muted)">Standby</div></div>
        <div class="tx-field"><div class="tx-label">Operation</div><div class="tx-val muted" id="tx-task">—</div></div>
      </div>

      <div class="progress-steps">
        <div class="step"><div class="step-bar" id="dot-0"></div><div class="step-label" id="lbl-0">INTAKE</div></div>
        <div class="step"><div class="step-bar" id="dot-1"></div><div class="step-label" id="lbl-1">INFERENCE</div></div>
        <div class="step"><div class="step-bar" id="dot-2"></div><div class="step-label" id="lbl-2">VERDICT</div></div>
      </div>
    </div>

    <!-- ── Episode State ────────────────────────────────────────── -->
    <div class="card card-episode">
      <h3><svg viewBox="0 0 24 24"><use href="#ic-scan"/></svg> Process State</h3>
      <div class="kv"><span class="k">Active Node</span><span class="v" id="ep-task" style="color:var(--accent-light)">—</span></div>
      <div class="kv"><span class="k">Nodes Checked</span><span class="v" id="ep-steps">—</span></div>
      <div class="kv"><span class="k">Done Flag</span><span class="v" id="ep-done">—</span></div>
      <div class="kv"><span class="k">Reward Score</span><span class="v" id="ep-rewards">—</span></div>
    </div>

    <!-- ── Diagnostics ──────────────────────────────────────────── -->
    <div class="card card-diagnostics">
      <h3><svg viewBox="0 0 24 24"><use href="#ic-check"/></svg> Diagnostics</h3>
      <div class="kv"><span class="k">API Link</span><span class="v" id="sys-api" style="color:var(--success)">CHECKING</span></div>
      <div class="kv"><span class="k">Model</span><span class="v" id="sys-model">—</span></div>
    </div>

    <!-- ── Verdict ──────────────────────────────────────────────── -->
    <div class="card card-verdict">
      <h3><svg viewBox="0 0 24 24"><use href="#ic-shield"/></svg> AI Verdict</h3>
      <div class="verdict-display" id="verdict-box">
        <div class="verdict-icon" id="verdict-icon"><svg viewBox="0 0 24 24"><use href="#ic-scan"/></svg></div>
        <div class="verdict-text" id="verdict-text">STANDBY</div>
        <div class="verdict-sub" id="verdict-sub">Awaiting data stream</div>
      </div>
    </div>

    <!-- ── Terminal Log (Toggleable) ────────────────────────────── -->
    <div class="card card-log">
      <div class="log-header">
        <h3><svg viewBox="0 0 24 24"><use href="#ic-term"/></svg> System Log</h3>
        <button class="btn-toggle" id="btn-term-toggle">
          View Terminal <span class="term-badge" id="term-badge"></span>
        </button>
      </div>
      <div class="console-wrapper" id="console-wrapper">
        <div class="console" id="console-log">
          <p><span class="ts">[BOOT]</span> <span class="msg info">MR.INSPECTOR Interface Online.</span></p>
        </div>
      </div>
    </div>

  </div>
</div>

<div id="toast-container"></div>

<script>
// ── Cinematic Intro Dismissal ──
window.addEventListener('load', () => {
  setTimeout(() => {
    const intro = document.getElementById('intro-overlay');
    if(intro) {
      intro.style.opacity = '0';
      intro.style.visibility = 'hidden';
      setTimeout(() => intro.remove(), 1000);
    }
  }, 3800);
});

// ── Terminal Toggle Logic ──
let isTermOpen = false;
document.getElementById('btn-term-toggle').addEventListener('click', (e) => {
  isTermOpen = !isTermOpen;
  const wrap = document.getElementById('console-wrapper');
  const badge = document.getElementById('term-badge');
  if(isTermOpen) {
    wrap.classList.add('open');
    e.currentTarget.innerHTML = `Hide Terminal <span class="term-badge" id="term-badge"></span>`;
    badge.classList.remove('active'); // clear notification
  } else {
    wrap.classList.remove('open');
    e.currentTarget.innerHTML = `View Terminal <span class="term-badge" id="term-badge"></span>`;
  }
});

// ── Utility Functions ──
function ts() { return new Date().toLocaleTimeString([], { hour12: false, hour:'2-digit', minute:'2-digit', second:'2-digit' }); }

function logEvent(msg, cls='info') {
  const el = document.getElementById('console-log');
  const p = document.createElement('p');
  p.innerHTML = `<span class="ts">[${ts()}]</span> <span class="msg ${cls}">${msg}</span>`;
  el.appendChild(p);
  el.scrollTop = el.scrollHeight; // Auto-scroll to bottom
  if(el.children.length > 50) el.removeChild(el.firstChild);

  // Ping the badge if terminal is closed
  if(!isTermOpen) {
    const badge = document.getElementById('term-badge');
    if(badge) badge.classList.add('active');
  }
}

function toast(msg) {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast'; t.textContent = msg; c.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateX(120%)'; setTimeout(() => t.remove(), 320); }, 4000);
}

function triggerFlash(isFraud) {
  const f = document.getElementById('flash-overlay');
  f.className = ''; void f.offsetWidth;
  f.className = isFraud ? 'flash-red' : 'flash-green';
  logEvent(isFraud ? '⚠️ ALERT: Target verified as FRAUD' : '✅ Target cleared as LEGITIMATE', isFraud ? 'err' : 'ok');
}

// ── Network Status ──
function updateNet() {
  const el = document.getElementById('net-status'), txt = document.getElementById('net-text');
  if(navigator.onLine) {
    el.className = 'status-badge live'; el.innerHTML = `<svg viewBox="0 0 24 24"><use href="#ic-check"/></svg> SECURE`;
  } else {
    el.className = 'status-badge offline'; el.innerHTML = `<svg viewBox="0 0 24 24"><use href="#ic-alert"/></svg> OFFLINE`;
    logEvent('Connection lost. Monitoring paused.', 'err'); toast('You are offline.');
  }
}
window.addEventListener('online', updateNet);
window.addEventListener('offline', updateNet);
updateNet();

// ── Data Polling & UI Updates ──
let lastTxId = null, lastStep = -1, lastTask = null, flashed = false;

async function sys() {
  if(!navigator.onLine) return;
  try {
    const r = await fetch('/system').then(res=>res.json());
    document.getElementById('sys-api').innerHTML = 'SECURE'; document.getElementById('sys-api').style.color = 'var(--success)';
    document.getElementById('sys-model').textContent = r.model_name || 'INSPECTOR_V1';
  } catch(e) { document.getElementById('sys-api').innerHTML = 'OFFLINE'; document.getElementById('sys-api').style.color = 'var(--danger)'; }
}

async function poll() {
  if(!navigator.onLine) return;
  try {
    const s = await fetch('/state').then(res=>res.json());
    
    const txId = s.transaction_id || null, isDone = s.done || false, rew = s.rewards || [], 
          steps = s.steps || 0, task = s.current_task || 'WAITING', act = s.episode_active || false, idx = s.task_index || 0;

    // Terminal Logging Logic (Updates on EVERY transaction step/task)
    if(txId && txId !== lastTxId) { lastTxId = txId; lastStep = -1; flashed = false; logEvent(`New Target Acquired: ${txId}`, 'info'); }
    if(act && steps !== lastStep && steps > 0) { lastStep = steps; logEvent(`Evaluating inference node... (Step ${steps})`, 'info'); }
    if(task !== lastTask && task !== 'WAITING' && task !== 'done') { lastTask = task; logEvent(`Process executing: ${task}`, 'warn'); }

    if(s.all_processed) {
      document.getElementById('btn-results').style.display = 'block';
      if(task !== 'done') logEvent(`Database scan complete. Awaiting final user command.`, 'info');
    }

    // Scanner
    const scan = document.getElementById('scanner');
    if(act && !isDone) scan.classList.add('active'); else scan.classList.remove('active');

    // Flash & Verdict Finalization
    if(isDone && s.predicted_label && lastStep !== -99) { lastStep = -99; triggerFlash(s.predicted_label === 'FRAUD'); }

    // Updates
    document.getElementById('tx-id-val').textContent = txId || '—';
    document.getElementById('tx-steps').textContent = steps;
    document.getElementById('tx-task').textContent = task;
    document.getElementById('ep-steps').textContent = steps;
    document.getElementById('ep-task').textContent = task;
    document.getElementById('ep-done').textContent = isDone ? 'TRUE' : 'FALSE';
    document.getElementById('ep-rewards').textContent = rew.length ? rew[0].toFixed(2) : '—';
    
    ['amount','merchant','category','location','card','time'].forEach(k => {
      if(s[k]) document.getElementById('tx-'+k).textContent = s[k];
    });

    const epVal = document.getElementById('tx-episode'), epWrap = document.getElementById('wrap-ep'), stWrap = document.getElementById('wrap-steps');
    if(act) { epVal.textContent = 'ACTIVE'; epVal.style.color = 'var(--accent-light)'; epWrap.className = 'tx-field highlight'; stWrap.className = 'tx-field highlight'; }
    else { epVal.textContent = 'STANDBY'; epVal.style.color = 'var(--text-muted)'; epWrap.className = 'tx-field'; stWrap.className = 'tx-field'; }

    // Verdict Box
    const vBox = document.getElementById('verdict-box'), vIc = document.getElementById('verdict-icon'), vTxt = document.getElementById('verdict-text'), vSub = document.getElementById('verdict-sub');
    if(isDone && s.predicted_label) {
      if(s.predicted_label === 'FRAUD') { vBox.className = 'verdict-display fraud'; vIc.innerHTML = '<svg viewBox="0 0 24 24"><use href="#ic-alert"/></svg>'; vTxt.textContent = 'FRAUD'; vSub.textContent = `Confidence: ${rew.length ? rew[0].toFixed(2) : '-'}`; }
      else { vBox.className = 'verdict-display legit'; vIc.innerHTML = '<svg viewBox="0 0 24 24"><use href="#ic-check"/></svg>'; vTxt.textContent = 'LEGITIMATE'; vSub.textContent = `Confidence: ${rew.length ? rew[0].toFixed(2) : '-'}`; }
    } else if(act) {
      vBox.className = 'verdict-display'; vIc.innerHTML = '<svg viewBox="0 0 24 24"><use href="#ic-scan"/></svg>'; vIc.style.color = 'var(--accent)';
      vTxt.textContent = 'DETECTING'; vTxt.style.color = 'var(--accent-light)'; vSub.textContent = `Step ${steps} running`;
    } else {
      vBox.className = 'verdict-display'; vIc.innerHTML = '<svg viewBox="0 0 24 24"><use href="#ic-scan"/></svg>'; vIc.style.color = '';
      vTxt.textContent = 'STANDBY'; vTxt.style.color = ''; vSub.textContent = 'Awaiting stream';
    }

    // Progress Dots
    for(let i=0; i<3; i++) {
      const d=document.getElementById('dot-'+i), l=document.getElementById('lbl-'+i);
      if(i<idx) { d.className='step-bar done'; l.className='step-label done'; }
      else if(i===idx && !isDone) { d.className='step-bar active'; l.className='step-label active'; }
      else { d.className='step-bar'; l.className='step-label'; }
    }
  } catch(e) {}
}

sys(); poll(); setInterval(sys, 5000); setInterval(poll, 1500);

async function showResults() {
  document.getElementById('results-modal').style.display = 'flex';
  try {
    const res = await fetch('/results').then(r => r.json());
    const hist = res.history || [];
    if(hist.length === 0) {
      document.getElementById('results-table-container').innerHTML = '<p>No records found.</p>';
      return;
    }
    let fraudHTML = '<table class="results-table"><tr><th>Trace ID</th><th>Actual</th><th>Match?</th></tr>';
    let legitHTML = '<table class="results-table"><tr><th>Trace ID</th><th>Actual</th><th>Match?</th></tr>';
    
    hist.forEach(h => {
      let match = (h.predicted_label === h.truth_label);
      let aClass = h.truth_label === 'FRAUD' ? 'label-fraud' : 'label-legit';
      let row = `<tr><td>${h.transaction_id}</td><td class="${aClass}">${h.truth_label || '-'}</td><td>${match ? '✅' : '❌'}</td></tr>`;
      
      if(h.predicted_label === 'FRAUD') {
        fraudHTML += row;
      } else {
        legitHTML += row;
      }
    });
    fraudHTML += '</table>';
    legitHTML += '</table>';
    
    let layout = `
      <div style="display: flex; gap: 2rem; justify-content: space-between;">
        <div style="flex: 1;">
          <h3 style="color: var(--danger); margin-bottom: 1rem;">Predicted Fraud</h3>
          ${fraudHTML}
        </div>
        <div style="flex: 1;">
          <h3 style="color: var(--success); margin-bottom: 1rem;">Predicted Legit</h3>
          ${legitHTML}
        </div>
      </div>
    `;
    document.getElementById('results-table-container').innerHTML = layout;
  } catch(e) {
    document.getElementById('results-table-container').innerHTML = '<p>Connection failed.</p>';
  }
}
function closeResults() {
  document.getElementById('results-modal').style.display = 'none';
}
</script>
</body>
</html>"""
    print("\n[DEBUG] 🌐 SERVING EMBEDDED FRONTEND PAGE!\n")
    return HTMLResponse(
        content=html_content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

def main():
    import uvicorn
    uvicorn.run("fraud_detect_env.server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
