"""
HIPAA-compliant PII / PHI scrubbing layer using Microsoft Presidio.

Detects and redacts 10+ PHI entity types:
  PERSON, PHONE_NUMBER, EMAIL_ADDRESS, US_SSN, US_DRIVER_LICENSE,
  CREDIT_CARD, IBAN_CODE, IP_ADDRESS, LOCATION, DATE_TIME,
  MEDICAL_LICENSE, NRP (National/Religious/Political groups)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

# ── PHI entity types covered ──────────────────────────────────────────────────
PHI_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "US_DRIVER_LICENSE",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "MEDICAL_LICENSE",
    "NRP",
]

REDACTION_PLACEHOLDER = "[REDACTED]"


@dataclass
class ScrubResult:
    original_text: str
    scrubbed_text: str
    entities_found: list[dict] = field(default_factory=list)
    entity_counts: dict = field(default_factory=dict)
    scrubbing_engine: str = "presidio"

    @property
    def phi_detected(self) -> bool:
        return len(self.entities_found) > 0

    @property
    def summary(self) -> str:
        if not self.phi_detected:
            return "No PHI detected."
        parts = [f"{k}: {v}" for k, v in self.entity_counts.items()]
        return "PHI redacted — " + ", ".join(parts)


# ── Presidio engine (lazy-loaded) ─────────────────────────────────────────────
_analyzer = None
_anonymizer = None


def _load_presidio():
    global _analyzer, _anonymizer
    if _analyzer is not None:
        return

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
        logger.info("Presidio engines loaded successfully")
    except Exception as e:
        logger.warning(f"Presidio not available ({e}) — using regex fallback scrubber")


# ── Regex fallback (when Presidio / spaCy model not installed) ────────────────
_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("US_SSN",        re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE_NUMBER",  re.compile(r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("EMAIL_ADDRESS", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")),
    ("IP_ADDRESS",    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    ("CREDIT_CARD",   re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("DATE_TIME",     re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        re.IGNORECASE,
    )),
]


def _regex_scrub(text: str) -> ScrubResult:
    scrubbed = text
    entities_found = []
    entity_counts: dict[str, int] = {}

    for entity_type, pattern in _REGEX_PATTERNS:
        matches = list(pattern.finditer(scrubbed))
        for m in matches:
            entities_found.append({
                "entity_type": entity_type,
                "text": m.group(),
                "start": m.start(),
                "end": m.end(),
                "score": 0.85,
            })
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        scrubbed = pattern.sub(f"[{entity_type}]", scrubbed)

    return ScrubResult(
        original_text=text,
        scrubbed_text=scrubbed,
        entities_found=entities_found,
        entity_counts=entity_counts,
        scrubbing_engine="regex-fallback",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def scrub_phi(
    text: str,
    entities: Optional[list[str]] = None,
    language: str = "en",
) -> ScrubResult:
    """
    Detect and redact PHI from clinical text.

    Args:
        text: Raw clinical note or document text.
        entities: Subset of PHI_ENTITIES to target (default: all).
        language: Language code for Presidio (default: 'en').

    Returns:
        ScrubResult with redacted text and entity metadata.
    """
    if not text or not text.strip():
        return ScrubResult(original_text=text, scrubbed_text=text)

    _load_presidio()
    target_entities = entities or PHI_ENTITIES

    if _analyzer is None:
        logger.debug("Using regex fallback scrubber")
        return _regex_scrub(text)

    try:
        from presidio_anonymizer.entities import OperatorConfig

        analyzer_results = _analyzer.analyze(
            text=text,
            entities=target_entities,
            language=language,
        )

        operators = {
            etype: OperatorConfig("replace", {"new_value": f"[{etype}]"})
            for etype in target_entities
        }

        anonymized = _anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators,
        )

        entities_found = [
            {
                "entity_type": r.entity_type,
                "text": text[r.start:r.end],
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
            }
            for r in analyzer_results
        ]

        entity_counts: dict[str, int] = {}
        for e in entities_found:
            t = e["entity_type"]
            entity_counts[t] = entity_counts.get(t, 0) + 1

        return ScrubResult(
            original_text=text,
            scrubbed_text=anonymized.text,
            entities_found=entities_found,
            entity_counts=entity_counts,
            scrubbing_engine="presidio",
        )

    except Exception as e:
        logger.error(f"Presidio scrubbing failed: {e} — falling back to regex")
        return _regex_scrub(text)


def scrub_dict(data: dict, keys_to_scrub: Optional[list[str]] = None) -> dict:
    """
    Recursively scrub PHI from string values in a dictionary.

    Args:
        data: Input dictionary (e.g., a parsed clinical note payload).
        keys_to_scrub: Specific keys to scrub; if None, scrubs all string values.

    Returns:
        New dictionary with PHI redacted from targeted fields.
    """
    result = {}
    for k, v in data.items():
        if isinstance(v, str) and (keys_to_scrub is None or k in keys_to_scrub):
            result[k] = scrub_phi(v).scrubbed_text
        elif isinstance(v, dict):
            result[k] = scrub_dict(v, keys_to_scrub)
        elif isinstance(v, list):
            result[k] = [
                scrub_phi(item).scrubbed_text if isinstance(item, str) else item
                for item in v
            ]
        else:
            result[k] = v
    return result
