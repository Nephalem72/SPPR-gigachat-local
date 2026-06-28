from sppr_colab.rag import build_grounded_context, get_rag_profile


def test_balanced_context_has_stable_source_ids_and_full_case_text() -> None:
    analysis = {
        "text": "Текст дела",
        "facts": {"roles": []},
        "ml_role_model": None,
        "legal_sources": [{"source": "УК РФ", "score": 0.7, "text": "Норма"}],
        "similar_cases": [
            {
                "case_number": "1-1/2024",
                "role_label": "организатор",
                "punishment_label": "лишение свободы",
                "score": 0.6,
                "fragment": "Фрагмент",
                "full_case": {"court": "Суд", "date": "2024-01-01", "article": "33", "text": "Полный текст"},
            }
        ],
    }

    context = build_grounded_context(analysis, get_rag_profile("balanced"))

    assert context["legal_sources"][0]["id"] == "L1"
    assert context["similar_cases"][0]["id"] == "C1"
    assert context["similar_cases"][0]["full_text"] == "Полный текст"


def test_unknown_profile_is_rejected() -> None:
    try:
        get_rag_profile("unknown")
    except ValueError as exc:
        assert "Allowed" in str(exc)
    else:
        raise AssertionError("Unknown profile must be rejected")
