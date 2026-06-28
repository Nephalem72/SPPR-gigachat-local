from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RagProfile:
    name: str
    legal_top_k: int
    case_top_k: int
    law_chars: int
    case_fragment_chars: int
    full_case_chars: int


RAG_PROFILES = {
    "fast": RagProfile("fast", 3, 3, 700, 500, 1000),
    "balanced": RagProfile("balanced", 5, 5, 1200, 700, 2200),
    "broad": RagProfile("broad", 8, 8, 1600, 900, 3500),
}


def get_rag_profile(name: str) -> RagProfile:
    try:
        return RAG_PROFILES[name]
    except KeyError as exc:
        allowed = ", ".join(RAG_PROFILES)
        raise ValueError(f"Unknown RAG profile '{name}'. Allowed: {allowed}") from exc


def profile_payload(profile: RagProfile) -> dict[str, Any]:
    return asdict(profile)


def build_grounded_context(analysis: dict[str, Any], profile: RagProfile) -> dict[str, Any]:
    legal_sources = []
    for index, item in enumerate(analysis["legal_sources"], 1):
        legal_sources.append(
            {
                "id": f"L{index}",
                "source": item.get("source", ""),
                "score": item.get("score", 0.0),
                "text": item.get("text", "")[: profile.law_chars],
            }
        )

    similar_cases = []
    for index, item in enumerate(analysis["similar_cases"], 1):
        full_case = item.get("full_case", {})
        similar_cases.append(
            {
                "id": f"C{index}",
                "case_number": item.get("case_number", ""),
                "court": full_case.get("court", ""),
                "date": full_case.get("date", ""),
                "article": full_case.get("article", ""),
                "role": item.get("role_label", ""),
                "punishment": item.get("punishment_label", ""),
                "score": item.get("score", 0.0),
                "fragment": item.get("fragment", "")[: profile.case_fragment_chars],
                "full_text": full_case.get("text", "")[: profile.full_case_chars],
            }
        )

    return {
        "case_text": analysis["text"],
        "facts": analysis["facts"],
        "ml_role_model": analysis["ml_role_model"],
        "legal_sources": legal_sources,
        "similar_cases": similar_cases,
        "rag_profile": profile_payload(profile),
    }
