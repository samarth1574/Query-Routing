"""The five specialised agents: Cards, Payments, Loans, Account and Fraud.

Each one maps the sub-intent from the NLP model to a short tool plan, then turns the tool
results and the retrieved policy passages into an outcome (resolved or not, facts, next
steps, and any reason a human should take over).
"""

from __future__ import annotations

import re

from .base import AgentOutcome, BaseAgent, ToolCall
from .tools import ToolResult

_SCAM = re.compile(r"\b(scam|scammed|scammer|tricked|safe account|investment scheme|romance|impersonat\w*)\b", re.I)


def _money(x, cur="£"):
    return f"{cur}{x:,.2f}"


class CardAgent(BaseAgent):
    name = "card_agent"
    category = "cards"
    description = "Card status, declines, PINs, delivery, activation and replacement."

    def plan(self, ctx, results, stage):
        if stage > 0:
            return []
        i = ctx.intent
        if i in ("card_swallowed",):
            return [ToolCall("block_card", {"customer_ref": ctx.customer_ref, "reason": "retained_by_atm"},
                             "an ATM kept the card, so it must be blocked and replaced")]
        if i in ("declined_card_payment", "declined_cash_withdrawal", "card_not_working", "contactless_not_working",
                 "virtual_card_not_working", "card_acceptance"):
            return [ToolCall("get_account_status", {"customer_ref": ctx.customer_ref}, "check card status and balance"),
                    ToolCall("list_recent_transactions", {"customer_ref": ctx.customer_ref, "status": "declined"},
                             "find the declined attempt")]
        return [ToolCall("get_account_status", {"customer_ref": ctx.customer_ref}, "look up the customer's cards")]

    def conclude(self, ctx, results, passages):
        facts, steps = [], []
        blocked = self.ok(results, "block_card")
        acct = self.ok(results, "get_account_status")
        declined = self.ok(results, "list_recent_transactions")
        if blocked:
            facts.append(f"Your card ending {blocked['card_last4']} is now blocked. {blocked['replacement']}")
        if acct:
            card = acct["cards"][0]
            facts.append(f"Your {card['type']} card ending {card['last4']} is {card['status']} "
                         f"and expires on {card['expires']}; available balance {_money(acct['balance'])}.")
            if card["status"] != "active":
                steps.append("Unblock or replace the card under Cards in the app.")
        if declined and declined["transactions"]:
            t = declined["transactions"][0]
            facts.append(f"The most recent declined payment was {_money(t['amount'])} at {t['merchant']} on {t['at'][:10]}.")
        facts += self.policy_facts(passages)
        steps.append("Reply here if the problem continues and we'll pass it to Card Services.")
        return AgentOutcome(self.name, True, "Card query answered with live card status and policy.", facts, steps)


class PaymentsAgent(BaseAgent):
    name = "payments_agent"
    category = "payments"
    description = "Pending, failed and reverted payments, transfers, refunds, fees and duplicate charges."

    STATUS_INTENTS = {"pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
                      "declined_transfer", "failed_transfer", "top_up_failed", "top_up_reverted", "reverted_card_payment?",
                      "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
                      "transfer_not_received_by_recipient", "Refund_not_showing_up", "request_refund", "cancel_transfer"}
    DISPUTE_INTENTS = {"transaction_charged_twice", "extra_charge_on_statement", "wrong_amount_of_cash_received"}

    def __init__(self, auto_dispute_limit: float = 100.0):
        self.auto_dispute_limit = auto_dispute_limit

    def plan(self, ctx, results, stage):
        amount = ctx.entities.get("amount")
        if stage == 0:
            if ctx.intent in self.STATUS_INTENTS:
                return [ToolCall("check_payment_status", {"customer_ref": ctx.customer_ref, "amount": amount},
                                 "find the payment and its status")]
            return [ToolCall("list_recent_transactions", {"customer_ref": ctx.customer_ref, "limit": 5},
                             "show recent activity")]
        if stage == 1 and ctx.intent in self.DISPUTE_INTENTS and amount and amount <= self.auto_dispute_limit:
            return [ToolCall("create_dispute", {"customer_ref": ctx.customer_ref, "reason": ctx.intent, "amount": amount},
                             f"duplicate/incorrect charge under the {_money(self.auto_dispute_limit)} auto-dispute limit")]
        return []

    def conclude(self, ctx, results, passages):
        facts, steps, reasons = [], [], []
        status = self.ok(results, "check_payment_status")
        recent = self.ok(results, "list_recent_transactions")
        dispute = self.ok(results, "create_dispute")
        if status:
            t = status["transaction"]
            facts.append(f"Your payment of {_money(t['amount'])} to {t['merchant']} on {t['at'][:10]} is "
                         f"{status['status']}. {status['explanation']}")
        if recent:
            listing = "; ".join(f"{_money(t['amount'])} {t['merchant']} ({t['status']})" for t in recent["transactions"][:3])
            facts.append(f"Your latest transactions: {listing}.")
        if dispute:
            facts.append(f"I've opened dispute {dispute['dispute_ref']}; expect an outcome within "
                         f"{dispute['expected_resolution_days']} working days.")
        amount = ctx.entities.get("amount")
        if ctx.intent in self.DISPUTE_INTENTS and not dispute:
            reasons.append("charge dispute above the automatic limit or without an amount"
                           if amount else "charge dispute needs the amount confirmed by an agent")
        facts += self.policy_facts(passages)
        steps.append("You can track this under Payments in the app.")
        return AgentOutcome(self.name, not reasons, "Payment query handled.", facts, steps, reasons)


class LoansAgent(BaseAgent):
    name = "loans_agent"
    category = "loans"
    description = "Loan eligibility, rates, EMIs, application status, repayments and closure."

    def __init__(self, auto_limit: float = 25000.0):
        self.auto_limit = auto_limit

    def plan(self, ctx, results, stage):
        if stage > 0:
            return []
        e = ctx.entities
        if ctx.intent in ("loan_application", "loan_interest_and_fees") or "eligib" in ctx.text.lower():
            args = {"customer_ref": ctx.customer_ref, "amount": e.get("amount"),
                    "tenure_months": e.get("tenure_months", 36), "loan_type": e.get("loan_type", "personal")}
            calls = [ToolCall("check_loan_eligibility", args, "indicative eligibility and EMI")]
            calls.append(ToolCall("get_loan_details", {"customer_ref": ctx.customer_ref}, "existing borrowing"))
            return calls
        return [ToolCall("get_loan_details", {"customer_ref": ctx.customer_ref}, "existing loans and due dates")]

    def conclude(self, ctx, results, passages):
        facts, steps, reasons = [], [], []
        elig = self.ok(results, "check_loan_eligibility")
        loans = self.ok(results, "get_loan_details")
        if elig:
            head = (f"For a {elig['loan_type']} loan of {_money(elig['requested_amount'])} over {elig['tenure_months']} "
                    f"months: indicative APR {elig['indicative_apr']}%, EMI about {_money(elig['monthly_emi'])}.")
            if elig["eligible"]:
                facts.append(head + " You look eligible (soft check only; this does not affect your credit score).")
                steps.append("Start the application under Loans → Apply in the app.")
            else:
                facts.append(head + " You're not eligible right now: " + "; ".join(elig["reasons"]) + ".")
                if elig["max_eligible_amount"] > 0:
                    steps.append(f"You may be eligible for up to {_money(elig['max_eligible_amount'])}.")
            if elig["requested_amount"] > self.auto_limit:
                reasons.append(f"loan request above {_money(self.auto_limit)} needs an underwriter")
        if loans is not None:
            if loans["loans"]:
                for ln in loans["loans"]:
                    facts.append(f"Loan {ln['id']} ({ln['type']}): {_money(ln['outstanding'])} outstanding, "
                                 f"EMI {_money(ln['emi'])}, next due {ln['next_due']}.")
            elif not elig:
                facts.append("You don't have an active loan with us.")
        if ctx.intent == "loan_repayment" and re.search(r"\b(can'?t|cannot|unable|struggl|hardship|miss)\w*", ctx.text, re.I):
            reasons.append("customer may be in financial difficulty")
        facts += self.policy_facts(passages)
        return AgentOutcome(self.name, not reasons, "Loan query answered.", facts, steps, reasons)


class AccountAgent(BaseAgent):
    name = "account_agent"
    category = "account"
    description = "Identity verification, passcodes, personal details, eligibility and closing accounts."

    def plan(self, ctx, results, stage):
        if stage > 0:
            return []
        i = ctx.intent
        if i == "passcode_forgotten":
            return [ToolCall("reset_passcode", {"customer_ref": ctx.customer_ref}, "send a secure reset link")]
        if i in ("verify_my_identity", "unable_to_verify_identity", "why_verify_identity", "verify_source_of_funds"):
            return [ToolCall("get_verification_requirements", {"customer_ref": ctx.customer_ref}, "KYC status and documents")]
        return [ToolCall("get_account_status", {"customer_ref": ctx.customer_ref}, "account status")]

    def conclude(self, ctx, results, passages):
        facts, steps, reasons = [], [], []
        reset = self.ok(results, "reset_passcode")
        kyc = self.ok(results, "get_verification_requirements")
        acct = self.ok(results, "get_account_status")
        if reset:
            facts.append(f"I've sent a passcode reset link to your {reset['channel']}; it expires in "
                         f"{reset['expires_minutes']} minutes.")
        if kyc:
            facts.append(f"Your verification status is {kyc['kyc_status']}. You'll need: {', '.join(kyc['documents'])}. "
                         f"Timing: {kyc['typical_time']}.")
            if kyc["kyc_status"] == "failed" or ctx.intent == "unable_to_verify_identity":
                reasons.append("identity verification failed or is stuck")
        if acct:
            facts.append(f"Your account is {acct['account_status']} (verification: {acct['kyc_status']}).")
        if ctx.intent == "terminate_account":
            reasons.append("account closure request goes to Account Services")
        facts += self.policy_facts(passages)
        return AgentOutcome(self.name, not reasons, "Account query handled.", facts, steps, reasons)


class FraudAgent(BaseAgent):
    name = "fraud_agent"
    category = "fraud"
    description = "Unrecognised transactions, compromised/lost/stolen cards and phones, scams and account takeover."

    CARD_LOSS = {"lost_or_stolen_card", "compromised_card", "lost_or_stolen_phone"}

    def __init__(self, high_value: float = 500.0):
        self.high_value = high_value

    def plan(self, ctx, results, stage):
        cr = ctx.customer_ref
        e = ctx.entities
        if stage == 0:
            if _SCAM.search(ctx.text):
                # Authorised push payment scams are never auto-actioned: gather evidence for the specialist.
                return [ToolCall("list_recent_transactions", {"customer_ref": cr, "limit": 5}, "evidence for the fraud team")]
            if ctx.intent in self.CARD_LOSS:
                # Protect first: blocking does not depend on anything else, so it runs alongside the lookup.
                return [ToolCall("block_card", {"customer_ref": cr, "reason": ctx.intent}, "stop further misuse now"),
                        ToolCall("list_recent_transactions", {"customer_ref": cr, "limit": 5}, "review recent activity")]
            return [ToolCall("verify_transaction", {"customer_ref": cr, "amount": e.get("amount"),
                                                    "merchant": e.get("merchant")}, "locate and risk-score the payment")]
        if stage == 1 and "verify_transaction" in results and results["verify_transaction"].ok:
            t = results["verify_transaction"].data["transaction"]
            # Independent actions → same stage → executed in parallel.
            return [ToolCall("block_card", {"customer_ref": cr, "reason": "unrecognised_transaction"}, "protect the card"),
                    ToolCall("create_dispute", {"customer_ref": cr, "transaction_id": t["id"], "amount": t["amount"],
                                                "reason": "unrecognised"}, "start the refund claim")]
        return []

    def conclude(self, ctx, results: dict[str, ToolResult], passages):
        facts, steps, reasons = [], [], []
        risk = "high"
        v = self.ok(results, "verify_transaction")
        blocked = self.ok(results, "block_card")
        dispute = self.ok(results, "create_dispute")
        recent = self.ok(results, "list_recent_transactions")
        if v:
            t = v["transaction"]
            risk = v["risk_level"]
            found = "I found the payment" if v["matched_on"] != "latest" else "Your most recent card payment is"
            facts.append(f"{found}: {_money(t['amount'])} at {t['merchant']} ({t['country']}) on {t['at'][:10]}, "
                         f"risk {v['risk_level']} ({v['risk_score']:.0%}).")
            if t["amount"] >= self.high_value:
                reasons.append(f"high-value unrecognised payment ({_money(t['amount'])})")
            if v["risk_level"] == "high":
                reasons.append("transaction risk score is high")
        if blocked:
            facts.append(f"Your card ending {blocked['card_last4']} is blocked so it can't be used again. "
                         f"{blocked['replacement']}")
        if dispute:
            facts.append(f"Dispute {dispute['dispute_ref']} is open; unauthorised payments are normally refunded within "
                         f"{dispute['expected_resolution_days']} working days.")
        if recent:
            facts.append("Please check these recent payments: " + "; ".join(
                f"{_money(t['amount'])} {t['merchant']} {t['at'][:10]}" for t in recent["transactions"][:3]) + ".")
        if _SCAM.search(ctx.text):
            reasons.append("possible scam / authorised push payment fraud — specialist review required")
        if ctx.intent == "lost_or_stolen_phone":
            steps.append("Log out of all devices from the web portal and ask your mobile operator to block the SIM.")
        steps.append("We will never ask for your PIN, passcode or one-time code.")
        facts += self.policy_facts(passages)
        return AgentOutcome(self.name, not reasons, "Fraud case contained.", facts, steps, reasons, risk)


def default_agents(policy: dict | None = None) -> dict[str, BaseAgent]:
    p = policy or {}
    return {
        "cards": CardAgent(),
        "payments": PaymentsAgent(auto_dispute_limit=float(p.get("auto_dispute_limit", 100))),
        "loans": LoansAgent(auto_limit=float(p.get("loan_auto_limit", 25000))),
        "account": AccountAgent(),
        "fraud": FraudAgent(high_value=float(p.get("fraud_high_value", 500))),
    }
