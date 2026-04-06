"""
inference.py — Standalone OpenEnv Agent Script

This is the AGENT. It is completely standalone:
  - NEVER imports from environment.py, server/app.py, or models.py
  - Talks to the environment ONLY via HTTP (localhost:7860)
  - Talks to the LLM ONLY via OpenAI client → HuggingFace router

Usage:
    python inference.py

Env vars:
    API_BASE_URL  — HuggingFace router base URL
    MODEL_NAME    — e.g. Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN      — HuggingFace API token (required, no default)
    ENV_BASE_URL  — environment server URL (default: http://localhost:7860)

Stdout format (exact, no deviation):
    [START] task=fraud_detection env=fraud_detect_env model=<model>
    [STEP]  step=1 action=FRAUD reward=1.00 done=false error=null
    [STEP]  step=2 action=ACCOUNT_TAKEOVER reward=1.00 done=false error=null
    [STEP]  step=3 action=<first 80 chars of plan> reward=0.80 done=true error=null
    [END]   success=true steps=3 score=0.933 rewards=1.00,1.00,0.80
"""

from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv
from openai import OpenAI

from fraud_detect_env.rewards_logger import save_episode

load_dotenv()

# ── Config (env vars only — no imports from this project) ───────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")      # ← NO default, required secret
ENV_BASE_URL = os.getenv("ENV_BASE_URL",  "http://localhost:7860")

# ── Prompts ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior bank fraud analyst AI with 20 years of experience.
You are highly accurate, decisive, and aggressive at catching fraud.
A missed fraud case causes real financial harm to the bank and customers.
Be precise. When in doubt, flag it as FRAUD."""

TASK_PROMPTS = {
    "classify": """TASK: CLASSIFY THIS TRANSACTION AS FRAUD OR L