from __future__ import annotations

from time import perf_counter
from typing import Any

from .extraction import extract_case_facts, normalize_text
from .llm import generate_chat_answer, generate_plain_chat_answer
from .models import predict_role_ml
from .rag import build_grounded_context, get_rag_profile
from .retrieval import fetch_full_cases, health_status, search_laws, search_similar_cases, warmup


def build_query(text: str, facts: dict[str, Any]) -> str:
    return " ".join(
        [
            text[:3000],
            " ".join(facts["articles_to_check"]),
            " ".join(item["label"] for item in facts["roles"]),
            " ".join(item["label"] for item in facts["punishments"]),
            "соучастие назначение наказания",
        ]
    )


def analyze_text(text: str, legal_top_k: int = 5, case_top_k: int = 5) -> dict[str, Any]:
    normalized = normalize_text(text)
    facts = extract_case_facts(normalized)
    query = build_query(normalized, facts)
    similar_cases = search_similar_cases(query, case_top_k)
    full_cases = fetch_full_cases(tuple(item["case_number"] for item in similar_cases))
    for item in similar_cases:
        item["full_case"] = full_cases.get(item["case_number"], {})
        item["full_case_found"] = bool(item["full_case"])
    return {
        "text": normalized,
        "facts": facts,
        "ml_role_model": predict_role_ml(normalized),
        "similar_cases": similar_cases,
        "legal_sources": search_laws(query, legal_top_k),
    }


def build_chat_context(
    text: str,
    rag_profile: str = "balanced",
    legal_top_k: int | None = None,
    case_top_k: int | None = None,
) -> dict[str, Any]:
    profile = get_rag_profile(rag_profile)
    payload = analyze_text(
        text,
        legal_top_k=legal_top_k or profile.legal_top_k,
        case_top_k=case_top_k or profile.case_top_k,
    )
    return build_grounded_context(payload, profile)


def answer_chat(
    text: str,
    question: str,
    history: list[dict[str, str]] | None = None,
    use_rag: bool = True,
    rag_profile: str = "balanced",
    legal_top_k: int | None = None,
    case_top_k: int | None = None,
    return_context: bool = False,
) -> dict[str, Any]:
    started = perf_counter()
    if not use_rag:
        response = generate_plain_chat_answer(text, question, history=history)
        total_seconds = perf_counter() - started
        response["metrics"].update(
            retrieval_seconds=0.0,
            total_seconds=round(total_seconds, 3),
            rag_enabled=False,
            rag_profile=None,
            legal_sources=0,
            similar_cases=0,
        )
        response["sources"] = []
        return response

    context = build_chat_context(
        text,
        rag_profile=rag_profile,
        legal_top_k=legal_top_k,
        case_top_k=case_top_k,
    )
    retrieval_seconds = perf_counter() - started
    response = generate_chat_answer(context, question, history=history)
    total_seconds = perf_counter() - started
    response["metrics"].update(
        retrieval_seconds=round(retrieval_seconds, 3),
        total_seconds=round(total_seconds, 3),
        rag_enabled=True,
        rag_profile=rag_profile,
        legal_sources=len(context["legal_sources"]),
        similar_cases=len(context["similar_cases"]),
    )
    response["sources"] = [
        {"id": item["id"], "source": item["source"], "score": item["score"]}
        for item in context["legal_sources"]
    ] + [
        {
            "id": item["id"],
            "case_number": item["case_number"],
            "court": item["court"],
            "date": item["date"],
            "score": item["score"],
        }
        for item in context["similar_cases"]
    ]
    if return_context:
        response["context"] = context
    return response


__all__ = ["analyze_text", "build_chat_context", "answer_chat", "health_status", "warmup"]
