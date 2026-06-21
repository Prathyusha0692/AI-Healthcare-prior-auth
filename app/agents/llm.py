"""
LLM factory — returns a real OpenAI ChatModel or a stub that generates
structured placeholder responses when no API key is configured.
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from app.config import get_settings

_settings = get_settings()


def _is_real_key(key: str) -> bool:
    return key.startswith("sk-") and "placeholder" not in key and len(key) > 20


# ── Stub LLM ──────────────────────────────────────────────────────────────────

class StubLLM:
    """Minimal stub that mimics .invoke() returning an AIMessage-like object."""

    class _Msg:
        def __init__(self, content: str):
            self.content = content

    RESPONSES: dict[str, str] = {
        "policy": json.dumps({
            "policy_summary": "Stub: The procedure requires prior authorization under the plan's specialist referral policy. Coverage is available for medically necessary procedures when documented appropriately.",
            "coverage_criteria": [
                "Documented medical necessity",
                "Primary care physician referral",
                "No available alternative treatments",
                "Pre-authorization form submitted 5 business days in advance",
            ],
        }),
        "clinical": json.dumps({
            "clinical_summary": "Stub: Patient presents with documented chronic condition requiring the requested procedure. History is consistent with standard treatment pathways.",
            "clinical_evidence": [
                "Documented diagnosis matching procedure indication",
                "Prior conservative treatment attempted",
                "Physician attestation of medical necessity",
            ],
            "clinical_flags": [
                "Lab results older than 90 days",
                "Missing specialist consultation note",
            ],
        }),
        "gap": json.dumps({
            "coverage_gaps": [
                "Specialist consultation note not provided",
                "Lab work predates 90-day requirement",
            ],
            "documentation_gaps": [
                "Recent lab panel (within 90 days)",
                "Signed specialist referral",
            ],
            "gap_severity": "medium",
        }),
        "recommendation": json.dumps({
            "authorization_decision": "pending_info",
            "confidence_score": 0.65,
            "recommendation_rationale": "Stub: Clinical documentation partially supports medical necessity, but two gaps must be resolved before a final decision can be made.",
            "required_actions": [
                "Submit specialist consultation note dated within 90 days",
                "Provide current lab panel results",
            ],
            "appeal_guidance": "If additional documentation is unavailable, submit a letter of medical necessity from the treating physician explaining why standard documentation cannot be obtained.",
        }),
    }

    def invoke(self, messages: list, agent_type: str = "recommendation") -> "_Msg":
        content = self.RESPONSES.get(agent_type, json.dumps({"result": "stub response"}))
        logger.debug(f"StubLLM returning response for agent_type={agent_type}")
        return self._Msg(content=content)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0):
    """Return a real ChatOpenAI or a StubLLM based on API key config."""
    if _is_real_key(_settings.openai_api_key):
        try:
            from langchain_openai import ChatOpenAI
            logger.info(f"Using ChatOpenAI model={_settings.openai_model}")
            return ChatOpenAI(
                model=_settings.openai_model,
                temperature=temperature,
                api_key=_settings.openai_api_key,
            )
        except Exception as e:
            logger.warning(f"ChatOpenAI init failed: {e}. Using StubLLM.")

    logger.warning("Using StubLLM — add a real OPENAI_API_KEY for production")
    return StubLLM()
