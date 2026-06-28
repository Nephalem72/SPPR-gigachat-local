from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import select

from sppr_colab import history_api
from sppr_colab.api import app
from sppr_colab.config import settings
from sppr_colab.database import User, get_engine, get_session_factory


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_users_are_isolated_and_history_is_persisted(tmp_path, monkeypatch) -> None:
    old_database_url = settings.database_url
    object.__setattr__(settings, "database_url", f"sqlite:///{(tmp_path / 'history.db').as_posix()}")
    get_session_factory.cache_clear()
    get_engine.cache_clear()
    monkeypatch.setattr(
        history_api,
        "answer_chat",
        lambda *args, **kwargs: {
            "answer": "Вывод подтверждается источником [L1].",
            "model": {"model_id": "test-model"},
            "sources": [{"id": "L1", "source": "УК РФ", "score": 0.9}],
            "metrics": {"total_seconds": 0.01},
            "citation_check": {"used": ["L1"], "invalid": [], "has_citations": True},
        },
    )

    try:
        with TestClient(app) as client:
            first = client.post(
                "/users",
                json={"display_name": "Первый", "username": "first", "password": "password1"},
            ).json()
            second = client.post(
                "/users",
                json={"display_name": "Второй", "username": "second", "password": "password2"},
            ).json()
            login = client.post("/login", json={"username": "first", "password": "password1"})
            assert login.status_code == 200
            first["api_token"] = login.json()["api_token"]
            created = client.post(
                "/conversations",
                headers=_auth(first["api_token"]),
                json={"title": "Дело", "case_text": "Обстоятельства дела", "rag_profile": "fast"},
            )
            assert created.status_code == 201
            conversation_id = created.json()["id"]

            hidden = client.get(f"/conversations/{conversation_id}", headers=_auth(second["api_token"]))
            assert hidden.status_code == 404

            answer = client.post(
                f"/conversations/{conversation_id}/messages",
                headers=_auth(first["api_token"]),
                json={"content": "Как квалифицировать роль?"},
            )
            assert answer.status_code == 200
            assert answer.json()["assistant_message"]["sources"][0]["id"] == "L1"

            history = client.get(f"/conversations/{conversation_id}", headers=_auth(first["api_token"])).json()
            assert [item["role"] for item in history["messages"]] == ["user", "assistant"]
            assert history["messages"][1]["model_id"] == "test-model"

            with get_session_factory()() as db:
                stored_user = db.scalar(select(User).where(User.id == first["id"]))
                assert stored_user is not None
                assert stored_user.token_hash != first["api_token"]
    finally:
        get_engine().dispose()
        get_session_factory.cache_clear()
        get_engine.cache_clear()
        object.__setattr__(settings, "database_url", old_database_url)


def test_conversation_message_can_disable_rag(tmp_path, monkeypatch) -> None:
    old_database_url = settings.database_url
    object.__setattr__(settings, "database_url", f"sqlite:///{(tmp_path / 'history.db').as_posix()}")
    get_session_factory.cache_clear()
    get_engine.cache_clear()
    captured: dict[str, object] = {}

    def fake_answer_chat(*args, **kwargs):
        captured.update(kwargs)
        return {
            "answer": "Обычный диалоговый ответ без поиска по базе.",
            "model": {"model_id": "test-model"},
            "sources": [],
            "metrics": {"total_seconds": 0.01, "retrieval_seconds": 0.0, "rag_enabled": False},
            "citation_check": {"used": [], "invalid": [], "has_citations": False},
        }

    monkeypatch.setattr(history_api, "answer_chat", fake_answer_chat)

    try:
        with TestClient(app) as client:
            user = client.post(
                "/users",
                json={"display_name": "Пользователь", "username": "person", "password": "password1"},
            ).json()
            created = client.post(
                "/conversations",
                headers=_auth(user["api_token"]),
                json={"title": "Дело", "case_text": "Обстоятельства дела", "rag_profile": "balanced"},
            )
            conversation_id = created.json()["id"]

            answer = client.post(
                f"/conversations/{conversation_id}/messages",
                headers=_auth(user["api_token"]),
                json={"content": "Просто обсуди позицию защиты.", "use_rag": False},
            )

            assert answer.status_code == 200
            assert captured["use_rag"] is False
            assert answer.json()["assistant_message"]["sources"] == []
    finally:
        get_engine().dispose()
        get_session_factory.cache_clear()
        get_engine.cache_clear()
        object.__setattr__(settings, "database_url", old_database_url)
