"""
inference.py — Standalone OpenEnv Agent Script

This is the AGENT. It is completely standalone:
  - NEVER imports from environment.py, server/app.py, or models.py
  - Talks to the environment ONLY via HTTP (localhost:7860)
  - Talks to the LLM ONLY via OpenAI client → HuggingFace router

Usage:
    python fraud_detect_env/inference.py

Env vars:
    API_BASE_URL  — HuggingFace router base URL
    MODEL_NAME    — e.g. Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN      — HuggingFace API token
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
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# ── Prompts (self-contained — no external import) ───────────────────────

SYSTEM_PROMPT = """You are an expert bank fraud analyst AI agent.
Be precise. Be accurate. A wrong decision causes real financial harm."""

TASK_PROMPTS = {
    "classify": """\
TASK 1 — CLASSIFY:
Read the transaction details carefully.
Reply with exactly one word: FRAUD or LEGIT.
No explanation. Just the word.

Transaction:
{txn_json}""",

    "identify_type": """\
TASK 2 — IDENTIFY FRAUD TYPE:
This transaction has been classified as FRAUD.
Identify the fraud type.
Reply with exactly one of these labels:
CARD_NOT_PRESENT, ACCOUNT_TAKEOVER, MONEY_LAUNDERING, IDENTITY_THEFT, PHISHING
No explanation. Just the label.

Transaction:
{txn_json}""",

    "action_plan": """\
TASK 3 — ACTION PLAN:
Fraud type: {fraud_type}

Write a complete action plan in this EXACT format:

Risk Level: HIGH/MEDIUM/LOW
Fraud Type: {fraud_type}
Recommended Action: <immediate action>
Next Steps:
  1. <step>
  2. <step>
  3. <step>
Do NOT: <what to avoid>

Transaction:
{txn_json}""",
}

VALID_LABELS      = {"FRAUD", "LEGIT"}
VALID_FRAUD_TYPES = {
    "CARD_NOT_PRESENT", "ACCOUNT_TAKEOVER", "MONEY_LAUNDERING",
    "IDENTITY_THEFT", "PHISHING",
}

# ── LLM Client ───────────────────────────────────────────────────────────

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def call_llm(user_prompt: str, max_tokens: int = 512) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()

# ── Post-processing helpers ──────────────────────────────────────────────

def extract_classification(raw: str) -> str:
    upper = raw.upper()
    for label in VALID_LABELS:
        if label in upper:
            return label
    return "LEGIT"  # safe fallback

def extract_fraud_type(raw: str) -> str:
    upper = raw.upper()
    for ft in VALID_FRAUD_TYPES:
        if ft in upper:
            return ft
    return "CARD_NOT_PRESENT"  # fallback

# ── Main Agent Loop ──────────────────────────────────────────────────────

def run():
    print(f"[START] task=fraud_detection env=fraud_detect_env model={MODEL_NAME}",
          flush=True)

    # ── RESET: start a new episode ──────────────────────────────────
    reset_resp = requests.post(f"{ENV_BASE_URL}/reset", timeout=10)
    reset_resp.raise_for_status()

    # /reset MUST return {"observation": {...}}
    observation = reset_resp.json()["observation"]
    
    if observation.get("all_processed"):
        print("[AGENT]  No more transactions to process! Exiting.")
        return False

    txn         = observation["transaction"]
    txn_id      = txn["transaction_id"]
    txn_json    = json.dumps(txn, indent=2)

    step_num      = 0
    all_rewards   = []
    detected_type = "UNKNOWN"

    # ── STEP LOOP ───────────────────────────────────────────────────
    while not observation.get("done", False):
        current_task = observation.get("current_task", "done")
        if current_task == "done":
            break

        error_str = "null"
        agent_reply = ""

        try:
            # Build prompt
            if current_task == "classify":
                prompt = TASK_PROMPTS["classify"].format(txn_json=txn_json)
                raw    = call_llm(prompt, max_tokens=10)
                agent_reply = extract_classification(raw)

            elif current_task == "identify_type":
                prompt = TASK_PROMPTS["identify_type"].format(txn_json=txn_json)
                raw    = call_llm(prompt, max_tokens=20)
                agent_reply   = extract_fraud_type(raw)
                detected_type = agent_reply

            elif current_task == "action_plan":
                prompt = TASK_PROMPTS["action_plan"].format(
                    txn_json=txn_json,
                    fraud_type=detected_type,
                )
                agent_reply = call_llm(prompt, max_tokens=512)

        except Exception as exc:  # noqa: BLE001
            agent_reply = ""
            error_str   = str(exc)[:120]

        # ── POST /step to environment ──────────────────────────────
        step_payload = {"task": current_task, "response": agent_reply}
        step_resp    = requests.post(
            f"{ENV_BASE_URL}/step", json=step_payload, timeout=30
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()

        reward      = step_data.get("reward", 0.0)
        done        = step_data.get("done", False)
        observation = step_data.get("observation", {})

        all_rewards.append(reward)
        step_num += 1

        # Truncate action to first 80 chars for the log line
        action_display = agent_reply.replace("\n", " ")[:80]

        # ANSI Colors
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        colored_action = action_display
        if "FRAUD" in action_display.upper():
            colored_action = f"{RED}{action_display}{RESET}"
        elif "LEGIT" in action_display.upper():
            colored_action = f"{GREEN}{action_display}{RESET}"

        print(
            f"[STEP] step={step_num} "
            f"action={colored_action} "
            f"reward={reward:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={error_str}",
            flush=True,
        )

        if done:
            break

    # ── END ─────────────────────────────────────────────────────────
    total_steps = step_num
    score       = round(sum(all_rewards) / len(all_rewards), 3) if all_rewards else 0.0
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    success     = score > 0.5

    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={total_steps} "
        f"score={score} "
        f"rewards={rewards_str}",
        flush=True,
    )

    # ── SAVE RESULTS ────────────────────────────────────────────────────
    save_episode(
        transaction_id=txn_id,
        rewards=all_rewards,
        steps=total_steps,
        score=score,
        model_name=MODEL_NAME,
    )
    return True


if __name__ == "__main__":
    try:
        run()
    except requests.exceptions.ConnectionError:
        print(
            f"[ERROR] Cannot connect to environment server at {ENV_BASE_URL}. "
            "Start it first with: uvicorn fraud_detect_env.server.app:app --port 7860",
            file=sys.stderr,
        )
        sys.exit(1)
