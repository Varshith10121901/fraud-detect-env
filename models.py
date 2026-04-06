"""
models.py — Pydantic data models for the Fraud Detection Agent.

Defines structured schemas for:
  - Transaction input
  - Classification result (FRAUD / LEGIT)
  - Fraud type identification
  - Action plan output
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class FraudLabel(str, Enum):
    FRAUD = "FRAUD"
    LEGIT = "LEGIT"


class FraudType(str, Enum):
    CARD_NOT_PRESENT = "CARD_NOT_PRESENT"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
    MONEY_LAUNDERING = "MONEY_LAUNDERING"
    IDENTITY_THEFT   = "IDENTITY_THEFT"
    PHISHING         = "PHISHING"


class RiskLevel(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


# ── Transaction Schema ───────────────────────────────────────────────────

class Transaction(BaseModel):
    """A single financial transaction to analyse."""

    transaction_id: str              = Field(..., description="Unique transaction identifier")
    timestamp: str                   = Field(..., description="ISO-8601 timestamp of the transaction")
    sender_account: str              = Field(..., description="Sender account number / ID")
    receiver_account: str            = Field(..., description="Receiver account number / ID")
    amount: float                    = Field(..., gt=0, description="Transaction amount in USD")
    currency: str                    = Field(default="USD", description="Currency code")
    transaction_type: str            = Field(..., description="e.g. WIRE, ACH, POS, ONLINE, ATM")
    merchant_name: Optional[str]     = Field(default=None, description="Merchant / payee name")
    merchant_category: Optional[str] = Field(default=None, description="MCC or category label")
    location: Optional[str]          = Field(default=None, description="City / country of the transaction")
    ip_address: Optional[str]        = Field(default=None, description="IP address if online txn")
    device_id: Optional[str]         = Field(default=None, description="Device fingerprint")
    is_international: bool           = Field(default=False, description="Cross-border flag")
    customer_age: Optional[int]      = Field(default=None, description="Age of the account holder")
    account_age_days: Optional[int]  = Field(default=None, description="Days since account was opened")
    previous_frauds: int             = Field(default=0, description="Number of prior fraud flags")
    notes: Optional[str]             = Field(default=None, description="Free-text notes / context")


# ── Response Schemas ─────────────────────────────────────────────────────

class ClassificationResult(BaseModel):
    """TASK 1 — binary classification."""
    transaction_id: str
    label: FraudLabel


class FraudTypeResult(BaseModel):
    """TASK 2 — fraud-type identification (only when label == FRAUD)."""
    transaction_id: str
    fraud_type: FraudType


class ActionPlan(BaseModel):
    """TASK 3 — structured remediation plan."""
    transaction_id: str
    risk_level: RiskLevel
    fraud_type: FraudType
    recommended_action: str
    next_steps: List[str]           = Field(..., min_length=3)
    do_not: str                     = Field(..., description="What to avoid doing")


class FullAnalysis(BaseModel):
    """Combined output of all three tasks for one transaction."""
    classification: ClassificationResult
    fraud_type: Optional[FraudTypeResult] = None
    action_plan: Optional[ActionPlan]     = None
