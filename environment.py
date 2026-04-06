"""
environment.py — OpenEnv-style environment for the Bank Fraud Detection Agent.

Orchestrates the three-task pipeline:
  TASK 1  →  Classify (FRAUD / LEGIT)
  TASK 2  →  Identify fraud type
  TASK 3  →  Generate action plan
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()  # loads from ../.env or .env

# ── System Prompts ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert bank fraud analyst AI agent.
You will be given a financial transaction and a specific task to complete.
Be precise. Be accurate. A wrong decision causes real financial harm."""

TASK1_PROMPT = """TASK 1 — CLASSIFY:
Read the transaction details carefully.
Reply with exactly one word: FRAUD or LEGIT.
No explanation. Just the word.

Transaction:
{transaction_json}"""

TASK2_PROMPT = """TASK 2 — IDENTIFY FRAUD TYPE:
The following transaction has been classified as FRAUD.
Identify the type of fraud.
Reply with exactly one of these labels:
CARD_NOT_PRESENT, ACCOUNT_TAKEOVER, MONEY_LAUNDERING, IDENTITY_THEFT, PHISHING
No explanation. Just the label.

Transaction:
{transaction_json}"""

TASK3_PROMPT = """TASK 3 — ACTION PLAN:
The following transaction has been classified as FRAUD.
Fraud Type: {fraud_type}

Write a complete action plan in this EXACT format (no markdown, no extra text):

Risk Level: HIGH/MEDIUM/LOW
Fraud Type: {fraud_type}
Recommended Action: <immediate action>
Next Steps:
  1. <step>
  2. <step>
  3. <step>
Do NOT: <what to avoid>

Transaction:
{transaction_json}"""


# ── Environment Config ───────────────────────────────────────────────────

class FraudDetectionEnv:
    """
    Lightweight environment wrapper that holds configuration
    and builds prompts for each task stage.
    """

    def __init__(self) -> None:
        self.api_base_url: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self.model_name: str   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        self.hf_token: str     = os.getenv("HF_TOKEN", "")

    # ── Prompt builders ──────────────────────────────────────────────

    @staticmethod
    def build_task1_prompt(txn: Dict[str, Any]) -> str:
        return TASK1_PROMPT.format(transaction_json=json.dumps(txn, indent=2))

    @staticmethod
    def build_task2_prompt(txn: Dict[str, Any]) -> str:
        return TASK2_PROMPT.format(transaction_json=json.dumps(txn, indent=2))

    @staticmethod
    def build_task3_prompt(txn: Dict[str, Any], fraud_type: str) -> str:
        return TASK3_PROMPT.format(
            fraud_type=fraud_type,
            transaction_json=json.dumps(txn, indent=2),
        )

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "api_base_url": self.api_base_url,
            "model_name": self.model_name,
            "hf_token": self.hf_token,
        }
