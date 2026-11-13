"""Simulated core-banking tools the agents can call.

Nothing here talks to a real bank. A small, deterministic in-memory "core banking"
store stands in for the systems a real deployment would call (card management,
payments ledger, lending decision engine, dispute case management).

Every tool:
- has a name, a description and the category it belongs to (shown at /api/v1/agents)
- is thread-safe, because the orchestrator calls independent tools in parallel
- can be given an artificial latency (BANKQ_TOOL_LATENCY_MS) to model a network hop,
  which is what makes parallel tool execution measurable in the benchmarks
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: dict = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {"tool": self.tool, "ok": self.ok, "data": self.data, "error": self.error,
                "latency_ms": self.latency_ms}


@dataclass
class Tool:
    name: str
    description: str
    category: str
    fn: Callable[..., dict]
    side_effect: bool = False  # True for tools that change state (block a card, open a dispute)


class ToolError(Exception):
    """Raised by a tool when the request is valid but the bank system refuses it."""


def _seed_for(customer_ref: str) -> int:
    return int(hashlib.sha256(customer_ref.encode()).hexdigest()[:8], 16)


class CoreBankingSimulator:
    """Deterministic fake bank: the same customer_ref always gets the same accounts, cards and history."""

    MERCHANTS = ["Tesco", "Amazon", "Uber", "Netflix", "Shell", "Zara", "Apple", "Deliveroo", "IKEA", "Ryanair"]

    def __init__(self, latency_ms: float = 0.0):
        self.latency_ms = latency_ms
        self._lock = threading.RLock()
        self._customers: dict[str, dict] = {}
        self._disputes: dict[str, dict] = {}
        self._dispute_seq = 0

    # ---------- state ----------

    def customer(self, customer_ref: str | None) -> dict:
        ref = customer_ref or "CUST-DEMO"
        with self._lock:
            if ref not in self._customers:
                self._customers[ref] = self._generate(ref)
            return self._customers[ref]

    def _generate(self, ref: str) -> dict:
        rng = random.Random(_seed_for(ref))
        now = datetime.now(UTC)
        card_last4 = f"{rng.randint(0, 9999):04d}"
        txns = []
        for i in range(12):
            status = rng.choices(["settled", "pending", "declined", "reverted"], [0.75, 0.12, 0.08, 0.05])[0]
            txns.append({
                "id": f"TXN-{_seed_for(ref + str(i)) % 10**8:08d}",
                "merchant": rng.choice(self.MERCHANTS),
                "amount": round(rng.uniform(3, 450), 2),
                "currency": "GBP",
                "status": status,
                "card_last4": card_last4,
                "at": (now - timedelta(hours=rng.randint(1, 24 * 20))).isoformat(timespec="seconds"),
                "country": rng.choice(["GB", "GB", "GB", "FR", "ES", "US"]),
            })
        txns.sort(key=lambda t: t["at"], reverse=True)
        return {
            "customer_ref": ref,
            "name_initial": rng.choice("ABCDEFGHJKLMNPRSTW"),
            "kyc_status": rng.choices(["verified", "pending", "failed"], [0.8, 0.15, 0.05])[0],
            "account_status": "active",
            "balance": round(rng.uniform(50, 12000), 2),
            "monthly_income": rng.choice([1800, 2400, 3200, 4100, 5500, 7800]),
            "existing_emi": rng.choice([0, 0, 150, 320, 600]),
            "credit_score": rng.randint(520, 860),
            "cards": [{"last4": card_last4, "type": rng.choice(["debit", "credit"]), "status": "active",
                       "expires": (now + timedelta(days=rng.randint(10, 900))).date().isoformat()}],
            "transactions": txns,
            "loans": [{"id": f"LN-{rng.randint(10000, 99999)}", "type": rng.choice(["personal", "car", "home"]),
                       "outstanding": round(rng.uniform(1000, 40000), 2), "emi": rng.choice([150, 320, 600]),
                       "next_due": (now + timedelta(days=rng.randint(1, 28))).date().isoformat()}]
            if rng.random() < 0.5 else [],
        }

    def _wait(self):
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)

    # ---------- tools ----------

    def get_account_status(self, customer_ref: str | None = None) -> dict:
        self._wait()
        c = self.customer(customer_ref)
        return {"customer_ref": c["customer_ref"], "account_status": c["account_status"], "kyc_status": c["kyc_status"],
                "balance": c["balance"], "cards": c["cards"]}

    def list_recent_transactions(self, customer_ref: str | None = None, limit: int = 5, status: str | None = None) -> dict:
        self._wait()
        txns = self.customer(customer_ref)["transactions"]
        if status:
            txns = [t for t in txns if t["status"] == status]
        return {"transactions": txns[:limit], "count": len(txns)}

    def verify_transaction(self, customer_ref: str | None = None, amount: float | None = None,
                           merchant: str | None = None) -> dict:
        """Find the transaction the customer is talking about and assess it."""
        self._wait()
        txns = self.customer(customer_ref)["transactions"]
        match = None
        if amount is not None:
            match = min(txns, key=lambda t: abs(t["amount"] - amount))
            if abs(match["amount"] - amount) > max(1.0, amount * 0.05):
                match = None
        if match is None and merchant:
            match = next((t for t in txns if t["merchant"].lower() == merchant.lower()), None)
        if match is None:
            match = txns[0]
        risk = 0.15
        if match["country"] != "GB":
            risk += 0.35
        if match["amount"] > 250:
            risk += 0.25
        if match["status"] == "pending":
            risk += 0.1
        return {"transaction": match, "risk_score": round(min(risk, 0.99), 2),
                "risk_level": "high" if risk >= 0.6 else "medium" if risk >= 0.35 else "low",
                "matched_on": "amount" if amount is not None else "merchant" if merchant else "latest"}

    def block_card(self, customer_ref: str | None = None, reason: str = "customer_request") -> dict:
        self._wait()
        with self._lock:
            c = self.customer(customer_ref)
            card = c["cards"][0]
            already = card["status"] == "blocked"
            card["status"] = "blocked"
            card["blocked_reason"] = reason
            card["blocked_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return {"card_last4": card["last4"], "status": "blocked", "already_blocked": already, "reason": reason,
                "replacement": "A replacement card will arrive in 3–5 working days."}

    def check_loan_eligibility(self, customer_ref: str | None = None, amount: float | None = None,
                               tenure_months: int = 36, loan_type: str = "personal") -> dict:
        self._wait()
        c = self.customer(customer_ref)
        rates = {"personal": 0.119, "car": 0.089, "home": 0.064, "student": 0.055}
        annual = rates.get(loan_type, 0.119)
        if c["credit_score"] >= 780:
            annual -= 0.015
        elif c["credit_score"] < 620:
            annual += 0.03
        r = annual / 12
        amount = amount or 10000
        emi = amount * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)
        dti = (c["existing_emi"] + emi) / c["monthly_income"]
        eligible = c["credit_score"] >= 600 and dti <= 0.45 and c["kyc_status"] == "verified"
        max_emi = max(0.45 * c["monthly_income"] - c["existing_emi"], 0)
        max_amount = max_emi * ((1 + r) ** tenure_months - 1) / (r * (1 + r) ** tenure_months)
        reasons = []
        if c["credit_score"] < 600:
            reasons.append("credit score below 600")
        if dti > 0.45:
            reasons.append(f"repayments would be {dti:.0%} of income (limit 45%)")
        if c["kyc_status"] != "verified":
            reasons.append("identity verification is not complete")
        return {"eligible": eligible, "loan_type": loan_type, "requested_amount": round(amount, 2),
                "tenure_months": tenure_months, "indicative_apr": round(annual * 100, 2), "monthly_emi": round(emi, 2),
                "debt_to_income": round(dti, 3), "max_eligible_amount": round(max_amount, -2), "reasons": reasons}

    def get_loan_details(self, customer_ref: str | None = None) -> dict:
        self._wait()
        return {"loans": self.customer(customer_ref)["loans"]}

    def check_payment_status(self, customer_ref: str | None = None, amount: float | None = None) -> dict:
        self._wait()
        txns = self.customer(customer_ref)["transactions"]
        candidates = [t for t in txns if t["status"] in ("pending", "declined", "reverted")] or txns
        if amount is not None:
            candidates = sorted(candidates, key=lambda t: abs(t["amount"] - amount))
        t = candidates[0]
        eta = {"pending": "Pending payments usually settle within 1–3 working days.",
               "declined": "The payment was declined and no money left the account.",
               "reverted": "The payment was reversed; funds are back in the account.",
               "settled": "The payment has settled."}[t["status"]]
        return {"transaction": t, "status": t["status"], "explanation": eta}

    def create_dispute(self, customer_ref: str | None = None, transaction_id: str | None = None,
                       reason: str = "unrecognised", amount: float | None = None) -> dict:
        self._wait()
        with self._lock:
            self._dispute_seq += 1
            ref = f"DSP-{self._dispute_seq:06d}"
            case = {"dispute_ref": ref, "customer_ref": customer_ref or "CUST-DEMO", "transaction_id": transaction_id,
                    "reason": reason, "amount": amount, "status": "open",
                    "opened_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "expected_resolution_days": 10 if reason == "unrecognised" else 15}
            self._disputes[ref] = case
        return case

    def reset_passcode(self, customer_ref: str | None = None) -> dict:
        self._wait()
        return {"reset_link_sent": True, "channel": "registered email", "expires_minutes": 30}

    def get_verification_requirements(self, customer_ref: str | None = None) -> dict:
        self._wait()
        c = self.customer(customer_ref)
        return {"kyc_status": c["kyc_status"],
                "documents": ["passport or driving licence", "a selfie in the app", "proof of address (if asked)"],
                "typical_time": "most checks complete within 10 minutes; manual checks take up to 2 working days"}


class ToolRegistry:
    """Names → callables, with timing and error capture so a failing tool never crashes an agent."""

    def __init__(self, bank: CoreBankingSimulator | None = None):
        self.bank = bank or CoreBankingSimulator()
        b = self.bank
        self._tools: dict[str, Tool] = {t.name: t for t in [
            Tool("get_account_status", "Account status, KYC state, balance and cards", "account", b.get_account_status),
            Tool("list_recent_transactions", "Most recent transactions, optionally filtered by status", "payments",
                 b.list_recent_transactions),
            Tool("verify_transaction", "Locate a transaction by amount or merchant and score its fraud risk", "fraud",
                 b.verify_transaction),
            Tool("block_card", "Block the customer's card and order a replacement", "cards", b.block_card, side_effect=True),
            Tool("check_loan_eligibility", "Indicative eligibility, APR and EMI for a loan amount and tenure", "loans",
                 b.check_loan_eligibility),
            Tool("get_loan_details", "Existing loans, outstanding balances and next due dates", "loans", b.get_loan_details),
            Tool("check_payment_status", "Status and explanation of a pending, declined or reverted payment", "payments",
                 b.check_payment_status),
            Tool("create_dispute", "Open a chargeback / dispute case for a transaction", "fraud", b.create_dispute,
                 side_effect=True),
            Tool("reset_passcode", "Send a passcode reset link to the registered email", "account", b.reset_passcode,
                 side_effect=True),
            Tool("get_verification_requirements", "Documents and timing for identity verification", "account",
                 b.get_verification_requirements),
        ]}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def describe(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "category": t.category, "side_effect": t.side_effect}
                for t in self._tools.values()]

    def call(self, name: str, **kwargs) -> ToolResult:
        start = time.perf_counter()
        try:
            data = self._tools[name].fn(**kwargs)
            return ToolResult(name, True, data, latency_ms=round((time.perf_counter() - start) * 1000, 2))
        except KeyError:
            return ToolResult(name, False, error=f"unknown tool {name!r}")
        except Exception as e:  # a tool failure must not take the agent down
            return ToolResult(name, False, error=f"{type(e).__name__}: {e}",
                              latency_ms=round((time.perf_counter() - start) * 1000, 2))
