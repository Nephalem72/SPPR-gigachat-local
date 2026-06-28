from __future__ import annotations

import re
from dataclasses import dataclass

from .dictionaries import (
    AGGRAVATING_PATTERNS,
    ARTICLE_PATTERNS,
    FORM_PATTERNS,
    MITIGATING_PATTERNS,
    PUNISHMENT_PATTERNS,
    ROLE_LABELS,
    ROLE_PATTERNS,
)


@dataclass(frozen=True)
class EvidenceMatch:
    label: str
    evidence: str
    confidence: str
    normalized: str | None = None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|;\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def first_sentence_with(regex: str, sentences: list[str]) -> str:
    for sentence in sentences:
        if re.search(regex, sentence, flags=re.I):
            return sentence[:900]
    return ""


def find_pattern_matches(text: str, patterns: dict[str, list[str]], labels: dict[str, str] | None = None) -> list[EvidenceMatch]:
    sentences = split_sentences(text)
    haystack = text.lower()
    matches: list[EvidenceMatch] = []
    for key, regexes in patterns.items():
        evidence = ""
        hit_count = 0
        for regex in regexes:
            if re.search(regex, haystack, flags=re.I):
                hit_count += 1
                if not evidence:
                    evidence = first_sentence_with(regex, sentences)
        if hit_count:
            matches.append(
                EvidenceMatch(
                    label=labels.get(key, key) if labels else key,
                    normalized=key,
                    evidence=evidence or "Найдено по ключевым формулировкам.",
                    confidence="высокая" if hit_count >= 2 else "средняя",
                )
            )
    return matches


def extract_case_facts(text: str) -> dict[str, object]:
    text = normalize_text(text)
    roles = find_pattern_matches(text, ROLE_PATTERNS, ROLE_LABELS)
    forms = find_pattern_matches(text, FORM_PATTERNS)
    mitigating = find_pattern_matches(text, MITIGATING_PATTERNS)
    aggravating = find_pattern_matches(text, AGGRAVATING_PATTERNS)
    punishments = find_pattern_matches(text, PUNISHMENT_PATTERNS)
    articles = [
        article
        for article, patterns in ARTICLE_PATTERNS.items()
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns)
    ]
    if roles and "ст. 33 УК РФ" not in articles:
        articles.insert(0, "ст. 33 УК РФ")
    if forms and "ст. 35 УК РФ" not in articles:
        articles.append("ст. 35 УК РФ")
    if punishments and "ст. 60 УК РФ" not in articles:
        articles.append("ст. 60 УК РФ")
    if aggravating and "ст. 63 УК РФ" not in articles:
        articles.append("ст. 63 УК РФ")
    if mitigating and "ст. 61 УК РФ" not in articles:
        articles.append("ст. 61 УК РФ")

    return {
        "roles": [item.__dict__ for item in roles],
        "forms": [item.__dict__ for item in forms],
        "mitigating": [item.__dict__ for item in mitigating],
        "aggravating": [item.__dict__ for item in aggravating],
        "punishments": [item.__dict__ for item in punishments],
        "articles_to_check": sorted(dict.fromkeys(articles)),
    }
