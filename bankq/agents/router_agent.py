"""The Router Agent: picks which specialist handles a query, and says why.

It starts from the NLP router's decision (category after business rules), then applies
agent-level safety logic:
- a strong secondary Fraud signal pulls the query to the Fraud agent (protect first),
- a sub-intent that clearly belongs to another category wins over a weak category guess.
"""

from __future__ import annotations

from ..taxonomy import INTENT_TO_CATEGORY
from .base import BaseAgent


class RouterAgent:
    def __init__(self, agents: dict[str, BaseAgent], fraud_secondary_threshold: float = 0.3,
                 intent_override_margin: float = 0.25):
        self.agents = agents
        self.fraud_secondary_threshold = fraud_secondary_threshold
        self.intent_override_margin = intent_override_margin

    def select(self, decision: dict) -> tuple[BaseAgent, list[str]]:
        reasons = []
        category = decision["category"]
        reasons.append(f"classifier → {category} ({decision['confidence']:.0%})")
        probs = decision.get("probabilities", {})

        if category != "fraud" and probs.get("fraud", 0) >= self.fraud_secondary_threshold:
            reasons.append(f"fraud probability {probs['fraud']:.0%} ≥ {self.fraud_secondary_threshold:.0%}: "
                           "routing to Fraud agent to protect the customer first")
            category = "fraud"
        else:
            intent = (decision.get("intent") or {})
            intent_cat = INTENT_TO_CATEGORY.get(intent.get("intent", ""))
            if (intent_cat and intent_cat != category and intent.get("confidence", 0) - decision["confidence"]
                    >= self.intent_override_margin):
                reasons.append(f"sub-intent {intent['intent']} ({intent['confidence']:.0%}) is more certain than "
                               f"the category; using the {intent_cat} agent")
                category = intent_cat

        agent = self.agents[category]
        reasons.append(f"selected {agent.name}")
        return agent, reasons
