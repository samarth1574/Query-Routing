"""Human-in-the-loop policy: decide when an automated resolution must be handed to a person.

Rules (configs/agent_policy.yaml):
- the classifier was unsure (needs_review)
- a sensitive business rule fired (vulnerable customer, formal complaint, account takeover)
- the specialist agent raised a reason (high-value fraud, scam, hardship, large loan, ...)
- a tool failed, so the agent could not act
- the answer is weakly grounded (best policy passage scored below `min_grounding_score`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SENSITIVE_RULES = {"vulnerable_customer", "formal_complaint", "account_takeover"}


@dataclass
class EscalationPolicy:
    min_grounding_score: float = 0.05
    escalate_on_review: bool = True
    escalate_on_tool_failure: bool = True
    sensitive_rules: set[str] = field(default_factory=lambda: set(SENSITIVE_RULES))
    agent_settings: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | None) -> EscalationPolicy:
        if not path or not Path(path).exists():
            return cls()
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        esc = cfg.get("escalation", {})
        return cls(
            min_grounding_score=float(esc.get("min_grounding_score", 0.05)),
            escalate_on_review=bool(esc.get("on_low_confidence", True)),
            escalate_on_tool_failure=bool(esc.get("on_tool_failure", True)),
            sensitive_rules=set(esc.get("sensitive_rules", SENSITIVE_RULES)),
            agent_settings=cfg.get("agents", {}),
        )

    def evaluate(self, decision: dict, outcome, tool_results: list, passages: list[dict]) -> list[str]:
        reasons = []
        if self.escalate_on_review and decision.get("needs_review"):
            reasons.append(f"classifier confidence {decision['confidence']:.0%} is below the review threshold")
        for r in decision.get("rules_applied", []):
            if r["rule"] in self.sensitive_rules:
                reasons.append(f"sensitive case: {r['rule'].replace('_', ' ')}")
        reasons += outcome.escalation_reasons
        if self.escalate_on_tool_failure:
            reasons += [f"tool {r.tool} failed: {r.error}" for r in tool_results if not r.ok]
        best = max((p["score"] for p in passages), default=0.0)
        if best < self.min_grounding_score:
            reasons.append(f"no policy passage matched well enough (best score {best:.2f})")
        return list(dict.fromkeys(reasons))
