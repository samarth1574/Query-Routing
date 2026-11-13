"""Turn an agent outcome into the customer-facing reply.

The default generator is template-based and only states facts that came from a tool result
or a retrieved policy passage, so every sentence is grounded and auditable. A generative
model can be plugged in behind the same `compose` interface (e.g. to rephrase the facts),
as long as it is constrained to the same facts.
"""

from __future__ import annotations

from typing import Protocol

from ..taxonomy import CATEGORIES, REVIEW_DEPARTMENT


class ResponseGenerator(Protocol):
    def compose(self, decision: dict, outcome, escalated: bool, reasons: list[str]) -> str: ...


class TemplateResponseGenerator:
    def compose(self, decision: dict, outcome, escalated: bool, reasons: list[str]) -> str:
        label = CATEGORIES[decision["category"]]["label"].lower()
        intent = (decision.get("intent") or {}).get("label")
        topic = f"{label} — {intent.lower()}" if intent else label
        if decision.get("needs_review"):
            lines = ["Thanks for your message. I want to be sure I understand it correctly before taking any action."]
            outcome.facts = []
            outcome.next_steps = []
        else:
            lines = [f"Thanks for getting in touch about your {topic}."]
        lines += [f"• {f}" for f in outcome.facts]
        if outcome.next_steps:
            lines.append("Next steps:")
            lines += [f"  – {s}" for s in outcome.next_steps]
        if escalated:
            team = REVIEW_DEPARTMENT if decision.get("needs_review") else CATEGORIES[decision["category"]]["department"]
            sla = decision.get("sla_hours")
            when = f" within {sla:g} hour{'s' if sla != 1 else ''}" if sla else ""
            lines.append(f"I've passed your case to our {team}, and a specialist will contact you{when}.")
            if outcome.facts and not decision.get("needs_review"):
                lines.append("Any protective steps above are already in place.")
        else:
            lines.append("Is there anything else I can help with?")
        return "\n".join(lines)
