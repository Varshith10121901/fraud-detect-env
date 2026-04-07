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

from rewards_logger import save_episode

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
    "classify": """You are a strict bank fraud detection AI. Your job is to flag suspicious transactions.

Transaction details:
{txn_json}

Fraud indicators to check:
- Unusually HIGH amount compared to merchant type
- Transaction at ODD HOURS (midnight to 5am)
- FOREIGN or unusual location for the account
- ONLINE/e-commerce with mismatched billing
- Rapid repeated transactions
- New merchant category for this card

Be AGGRESSIVE in flagging fraud. When in doubt, say FRAUD.

Respond with EXACTLY one word only — either FRAUD or LEGIT. No explanation.""",

    "identify_type": """You are a fraud analyst. This transaction was flagged as FRAUD.

Transaction:
{txn_json}

Choose EXACTLY one fraud type from this list:
- CARD_NOT_PRESENT → online/remote purchase fraud
- ACCOUNT_TAKEOVER → stolen credentials, login from new device
- MONEY_LAUNDERING → structuring, round amounts, layering
- IDENTITY_THEFT → new account, address mismatch
- PHISHING → social engineering, fake merchant

Respond with EXACTLY one of those 5 options. No other words.""",

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

# ── LLM Client ────────────────────────────────────────────────────────────

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

# ── Post-processing helpers ───────────────────────────────────────────────

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

TASK_SEQUENCE = ["classify", "identify_type", "action_plan"]

def run() -> bool:
    print(f"[START] task=fraud_detection env=fraud_detect_env model={MODEL_NAME}",
          flush=True)

    # ── RESET: start a new episode ─────────────────────────────────────
    reset_resp = requests.post(f"{ENV_BASE_URL}/reset", timeout=10)
    reset_resp.raise_for_status()

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

    # ── STEP LOOP ────────────────────────────────────────────────────────
    while not observation.get("done", False):
        current_task = observation.get("current_task", "done")
        if current_task == "done":
            break

        error_str   = "null"
        agent_reply = ""

        try:
            if current_task == "classify":
                prompt      = TASK_PROMPTS["classify"].format(txn_json=txn_json)
                raw         = call_llm(prompt, max_tokens=10)
                agent_reply = extract_classification(raw)

            elif current_task == "identify_type":
                prompt        = TASK_PROMPTS["identify_type"].format(txn_json=txn_json)
                raw           = call_llm(prompt, max_tokens=20)
                agent_reply   = extract_fraud_type(raw)
                detected_type = agent_reply

            elif current_task == "action_plan":
                prompt      = TASK_PROMPTS["action_plan"].format(
                    txn_json=txn_json,
                    fraud_type=detected_type,
                )
                agent_reply = call_llm(prompt, max_tokens=512)

        except Exception as exc:
            agent_reply = ""
            error_str   = str(exc)[:120]

        # ── POST /step to environment ───────────────────────────────────
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

        action_display = agent_reply.replace("\n", " ")[:80]

        RED   = "\033[91m"
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

    # ── END ──────────────────────────────────────────────────────────────
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

    try:
        save_episode(
            transaction_id=txn_id,
            rewards=all_rewards,
            steps=total_steps,
            score=score,
            model_name=MODEL_NAME,
        )
    except Exception:
        pass  # rewards logging is optional, never crash the agent

    return True


if __name__ == "__main__":
    try:
        run()
    except requests.exceptions.ConnectionError:
        print(
            f"[ERROR] Cannot connect to environment server at {ENV_BASE_URL}. "
            "Start it first with: uvicorn server.app:app --port 7860",
            file=sys.stderr,
        )
        sys.exit(1)