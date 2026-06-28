from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import joblib

from .config import settings
from .dictionaries import ROLE_LABELS


@lru_cache(maxsize=1)
def load_role_model() -> tuple[Any | None, Any | None, str]:
    if not settings.role_model_path.exists() or not settings.role_vectorizer_path.exists():
        return None, None, "role_model.pkl/vectorizer.pkl не найдены"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vectorizer = joblib.load(settings.role_vectorizer_path)
            model = joblib.load(settings.role_model_path)
        warning_text = "; ".join(str(item.message) for item in caught[:3])
        return vectorizer, model, warning_text
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def predict_role_ml(text: str) -> dict[str, Any]:
    vectorizer, model, warning_text = load_role_model()
    if vectorizer is None or model is None:
        return {"available": False, "message": warning_text}
    try:
        features = vectorizer.transform([text])
        predicted = str(model.predict(features)[0])
        return {
            "available": True,
            "predicted_role": ROLE_LABELS.get(predicted, predicted),
            "normalized_role": predicted,
            "warning": warning_text,
        }
    except Exception as exc:
        return {"available": False, "message": f"{type(exc).__name__}: {exc}"}
