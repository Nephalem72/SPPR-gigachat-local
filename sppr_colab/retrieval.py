from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import faiss
import joblib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import settings
from .dictionaries import PUNISHMENT_LABELS, ROLE_LABELS
from .extraction import extract_case_facts, normalize_text


@dataclass
class TfidfIndex:
    frame: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: Any
    text_column: str


@dataclass
class LegacyEmbeddingIndex:
    frame: pd.DataFrame
    embeddings: np.ndarray
    index: Any
    model_name: str


def _build_tfidf(frame: pd.DataFrame, text_column: str, max_features: int = 50_000) -> TfidfIndex:
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=max_features, min_df=1)
    matrix = vectorizer.fit_transform(frame[text_column].fillna("").astype(str))
    return TfidfIndex(frame=frame.reset_index(drop=True), vectorizer=vectorizer, matrix=matrix, text_column=text_column)


@lru_cache(maxsize=1)
def load_case_encoder() -> SentenceTransformer:
    return SentenceTransformer(settings.legacy_case_encoder, device="cpu")


@lru_cache(maxsize=1)
def load_laws_index() -> TfidfIndex | None:
    if not settings.laws_path.exists():
        return None
    frame = pd.read_parquet(settings.laws_path, columns=["source", "text"]).dropna(subset=["text"]).copy()
    frame["text"] = frame["text"].astype(str).map(normalize_text)
    frame = frame[frame["text"].str.len() > 40].reset_index(drop=True)
    return _build_tfidf(frame, "text", max_features=60_000)


@lru_cache(maxsize=1)
def load_case_index() -> LegacyEmbeddingIndex | None:
    if not settings.final_dataset_path.exists() or not settings.embeddings_path.exists() or not settings.faiss_index_path.exists():
        return None
    frame = pd.read_parquet(settings.final_dataset_path).copy()
    embeddings = np.asarray(joblib.load(settings.embeddings_path), dtype=np.float32)
    index = faiss.read_index(str(settings.faiss_index_path))
    if len(frame) != len(embeddings) or index.ntotal != len(frame):
        return None
    return LegacyEmbeddingIndex(frame=frame.reset_index(drop=True), embeddings=embeddings, index=index, model_name=settings.legacy_case_encoder)


def warmup() -> dict[str, Any]:
    load_laws_index()
    load_case_index()
    load_case_encoder()
    return health_status()


def health_status() -> dict[str, Any]:
    return {
        "drive_root": str(settings.drive_root),
        "data_dir": str(settings.data_dir),
        "laws_exists": settings.laws_path.exists(),
        "final_dataset_exists": settings.final_dataset_path.exists(),
        "cases_exists": settings.cases_path.exists(),
        "embeddings_exists": settings.embeddings_path.exists(),
        "faiss_exists": settings.faiss_index_path.exists(),
        "role_model_exists": settings.role_model_path.exists(),
        "vectorizer_exists": settings.role_vectorizer_path.exists(),
        "case_encoder": settings.legacy_case_encoder,
    }


def search_laws(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    index = load_laws_index()
    if index is None or not query.strip():
        return []
    query_vector = index.vectorizer.transform([normalize_text(query)])
    scores = cosine_similarity(query_vector, index.matrix).ravel()
    top_indices = scores.argsort()[::-1][:top_k]
    results: list[dict[str, Any]] = []
    for idx in top_indices:
        score = float(scores[int(idx)])
        if score <= 0:
            continue
        row = index.frame.iloc[int(idx)]
        results.append(
            {
                "score": round(score, 4),
                "source": str(row.get("source", "")),
                "text": str(row.get("text", ""))[:1200],
            }
        )
    return results


def _case_rerank_bonus(query_facts: dict[str, Any], row: pd.Series) -> float:
    text = normalize_text(f"{row.get('fragment', '')} {row.get('context', '')}")
    row_facts = extract_case_facts(text)
    bonus = 0.0
    query_roles = {item.get("normalized") for item in query_facts.get("roles", []) if item.get("normalized")}
    row_role = str(row.get("role", "")).strip()
    if query_roles and row_role in query_roles:
        bonus += 0.08
    query_forms = {item.get("label", "").lower() for item in query_facts.get("forms", []) if item.get("label")}
    row_forms = {item.get("label", "").lower() for item in row_facts.get("forms", []) if item.get("label")}
    if query_forms and row_forms and query_forms & row_forms:
        bonus += 0.04
    return bonus


def search_similar_cases(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    index = load_case_index()
    if index is None or not query.strip():
        return []
    query_facts = extract_case_facts(query)
    vector = load_case_encoder().encode(
        [normalize_text(query)],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    distances, indices = index.index.search(vector, max(top_k * 8, 20))
    seen_case_numbers: set[str] = set()
    results: list[dict[str, Any]] = []
    for idx_row, distance in zip(indices[0].tolist(), distances[0].tolist(), strict=False):
        if idx_row < 0:
            continue
        row = index.frame.iloc[int(idx_row)]
        case_number = str(row.get("case_number", ""))
        if not case_number or case_number in seen_case_numbers:
            continue
        seen_case_numbers.add(case_number)
        base_score = 1.0 / (1.0 + max(float(distance), 0.0))
        score = min(base_score + _case_rerank_bonus(query_facts, row), 0.99)
        results.append(
            {
                "score": round(score, 4),
                "base_score": round(base_score, 4),
                "case_number": case_number,
                "person": str(row.get("person", "")),
                "role": str(row.get("role", "")),
                "role_label": ROLE_LABELS.get(str(row.get("role", "")), str(row.get("role", ""))),
                "punishment_type": str(row.get("punishment_type", "")),
                "punishment_label": PUNISHMENT_LABELS.get(str(row.get("punishment_type", "")), str(row.get("punishment_type", ""))),
                "punishment_value": str(row.get("punishment_value", "")),
                "fragment": str(row.get("fragment", "")),
                "context": str(row.get("context", "")),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


@lru_cache(maxsize=64)
def fetch_full_cases(case_numbers: tuple[str, ...]) -> dict[str, dict[str, str]]:
    if not settings.cases_path.exists() or not case_numbers:
        return {}
    unique_numbers = tuple(dict.fromkeys(str(item) for item in case_numbers if str(item).strip()))
    if not unique_numbers:
        return {}
    dataset = ds.dataset(settings.cases_path, format="parquet")
    table = dataset.to_table(
        columns=["case_number", "court", "date", "article", "instance", "source_file", "text"],
        filter=pc.field("case_number").isin(list(unique_numbers)),
    )
    result: dict[str, dict[str, str]] = {}
    for row in table.to_pylist():
        number = str(row.get("case_number", ""))
        if number and number not in result:
            result[number] = {key: str(value or "") for key, value in row.items()}
    return result
