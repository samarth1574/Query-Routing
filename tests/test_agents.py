"""Agent-layer tests: entities, tools, RAG, specialists, router agent and escalation, with the real model."""

import pytest

from bankq.agents.entities import extract_entities
from bankq.agents.orchestrator import ResolutionOrchestrator
from bankq.agents.rag import KnowledgeBase
from bankq.agents.router_agent import RouterAgent
from bankq.agents.specialists import default_agents
from bankq.agents.tools import CoreBankingSimulator, ToolRegistry
from bankq.config import ROOT


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load(ROOT / "knowledge_base")


@pytest.fixture(scope="module")
def orch():
    from bankq.api.app import build_router
    o = ResolutionOrchestrator.from_paths(build_router(), ROOT / "knowledge_base", ROOT / "configs" / "agent_policy.yaml")
    yield o
    o.close()


# ---------- entities ----------

@pytest.mark.parametrize("text,expected", [
    ("I don't recognise £45.20 at Amazon", {"amount": 45.2, "currency": "GBP", "merchant": "Amazon"}),
    ("car loan of £15,000 over 4 years", {"amount": 15000, "tenure_months": 48, "loan_type": "car"}),
    ("I need 20k for a mortgage over 240 months", {"tenure_months": 240, "loan_type": "home"}),
    ("charged twice 30 pounds", {"amount": 30.0, "currency": "GBP"}),
])
def test_extract_entities(text, expected):
    got = extract_entities(text)
    for k, v in expected.items():
        assert got[k] == v


# ---------- tools ----------

def test_simulator_is_deterministic_per_customer():
    a, b = CoreBankingSimulator(), CoreBankingSimulator()
    assert a.customer("CUST-9")["cards"] == b.customer("CUST-9")["cards"]


def test_block_card_changes_state_and_is_idempotent():
    tools = ToolRegistry()
    first = tools.call("block_card", customer_ref="C1")
    second = tools.call("block_card", customer_ref="C1")
    assert first.ok and not first.data["already_blocked"]
    assert second.data["already_blocked"]
    assert tools.call("get_account_status", customer_ref="C1").data["cards"][0]["status"] == "blocked"


def test_tool_failure_is_captured_not_raised():
    tools = ToolRegistry()
    r = tools.call("check_loan_eligibility", customer_ref="C1", tenure_months="bad")
    assert not r.ok and r.error
    assert not tools.call("no_such_tool").ok


def test_loan_eligibility_maths():
    d = ToolRegistry().call("check_loan_eligibility", customer_ref="C2", amount=12000, tenure_months=24).data
    assert d["monthly_emi"] * 24 > 12000  # interest is charged
    assert 0 < d["debt_to_income"] < 5


# ---------- RAG ----------

def test_rag_finds_the_right_policy(kb):
    assert kb.search("my card was skimmed at an atm", "fraud")[0]["heading"].startswith("Compromised")
    assert kb.search("how long does a transfer take")[0]["heading"] == "Transfer timing"
    assert all(p["category"] == "loans" for p in kb.search("interest rate", "loans"))


def test_rag_batch_matches_single(kb):
    qs = ["pin blocked", "close my account", "refund not arrived"]
    batch = kb.search_many(qs, k=1)
    for q, b in zip(qs, batch):
        assert b[0]["id"] == kb.search(q, k=1)[0]["id"]


# ---------- router agent ----------

def _decision(category, conf, probs=None, intent=None):
    return {"category": category, "confidence": conf, "probabilities": probs or {category: conf},
            "intent": intent}


def test_router_agent_prefers_fraud_when_fraud_probability_is_high():
    agent, reasons = RouterAgent(default_agents()).select(_decision("cards", 0.55, {"cards": 0.55, "fraud": 0.4}))
    assert agent.name == "fraud_agent" and any("Fraud agent" in r for r in reasons)


def test_router_agent_follows_classifier_normally():
    agent, _ = RouterAgent(default_agents()).select(_decision("loans", 0.95, {"loans": 0.95, "fraud": 0.01}))
    assert agent.name == "loans_agent"


# ---------- end to end with the real model ----------

def test_unrecognised_payment_blocks_card_and_opens_dispute_in_parallel(orch):
    r = orch.resolve("I don't recognise a payment of £45 at Amazon", "CUST-T1")
    assert r["agent"] == "fraud_agent"
    tools = [c["tool"] for s in r["actions"] for c in s["calls"]]
    assert tools[0] == "verify_transaction" and {"block_card", "create_dispute"} <= set(tools)
    stage2 = r["actions"][1]
    assert stage2["parallel"] and len(stage2["calls"]) == 2
    assert "Dispute DSP-" in r["reply"]
    assert r["knowledge"] and r["knowledge"][0]["category"] == "fraud"


def test_scam_is_always_escalated_without_automatic_actions(orch):
    r = orch.resolve("I was scammed into moving money to a safe account", "CUST-T2")
    assert r["status"] == "escalated"
    tools = {c["tool"] for s in r["actions"] for c in s["calls"]}
    assert "create_dispute" not in tools and "block_card" not in tools


def test_vulnerable_customer_goes_to_a_human(orch):
    r = orch.resolve("My husband passed away and I need help with his loan", "CUST-T3")
    assert r["status"] == "escalated"
    assert any("vulnerable" in x for x in r["escalation"]["reasons"])


def test_low_confidence_takes_no_action(orch):
    r = orch.resolve("hello", "CUST-T4")
    assert r["status"] == "escalated" and r["actions"] == []
    assert r["escalation"]["department"].startswith("General Support")


def test_simple_queries_are_resolved_automatically(orch):
    for q in ["I forgot my passcode", "Am I eligible for a car loan of £15,000 over 4 years?",
              "my transfer of £200 is still pending"]:
        r = orch.resolve(q, "CUST-T5")
        assert r["status"] == "resolved", (q, r["escalation"])


def test_large_loan_needs_an_underwriter(orch):
    r = orch.resolve("I'd like a personal loan of £40,000 over 5 years, am I eligible?", "CUST-T6")
    assert r["agent"] == "loans_agent" and r["status"] == "escalated"


def test_pii_never_reaches_the_reply(orch):
    r = orch.resolve("My card 4539 1488 0343 6467 was skimmed", "CUST-T7")
    assert "4539" not in r["reply"] and r["pii_redacted"] == ["CARD"]
