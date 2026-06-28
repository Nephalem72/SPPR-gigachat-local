from __future__ import annotations

from typing import Any

import gradio as gr
import requests

from .config import settings


CSS = """
.gradio-container { max-width: 1500px !important; }
#app-title h1 { font-size: 1.55rem; margin: 0; letter-spacing: 0; }
#app-title p { color: var(--body-text-color-subdued); margin-top: .25rem; }
#chat { min-height: 760px; }
#status { min-height: 1.5rem; }
"""


def _headers(token: str = "") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request(method: str, path: str, token: str = "", **kwargs: Any) -> Any:
    try:
        response = requests.request(
            method,
            f"{settings.api_url.rstrip('/')}{path}",
            headers={**_headers(token), **kwargs.pop("headers", {})},
            timeout=kwargs.pop("timeout", 900),
            **kwargs,
        )
    except requests.RequestException as exc:
        raise gr.Error(f"Сервер недоступен: {exc}") from exc
    if response.ok:
        return response.json() if response.content else None
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise gr.Error(f"Ошибка сервера ({response.status_code}): {detail}")


def _conversation_choices(token: str) -> list[tuple[str, str]]:
    if not token:
        return []
    payload = _request("GET", "/conversations", token, params={"limit": 100})
    return [(item["title"], item["id"]) for item in payload["items"]]


def _sources_view(sources: list[dict[str, Any]] | None) -> tuple[list[list[Any]], list[tuple[str, str]]]:
    rows: list[list[Any]] = []
    case_choices: list[tuple[str, str]] = []
    for item in sources or []:
        score = f"{float(item.get('score', 0.0)):.1%}"
        if item.get("case_number"):
            name = f"Дело {item['case_number']}"
            details = " · ".join(value for value in [item.get("court", ""), item.get("date", "")] if value)
            rows.append([item.get("id", ""), "Судебное дело", name, details, score])
            case_choices.append((f"{item.get('id', '')} · {name}", item["case_number"]))
        else:
            rows.append([item.get("id", ""), "Правовой материал", item.get("source", ""), "", score])
    return rows, case_choices


def _confidence_percent(confidence: str) -> str:
    value = {"высокая": 85, "средняя": 60, "низкая": 35}.get((confidence or "").lower(), 50)
    return f"{value}%"


def _analysis_views(payload: dict[str, Any] | None) -> tuple[str, list[list[Any]], list[list[Any]], list[list[Any]], list[tuple[str, str]]]:
    if not payload:
        return "", [], [], [], []
    facts = payload.get("facts") or {}
    roles = facts.get("roles") or []
    forms = facts.get("forms") or []
    articles = facts.get("articles_to_check") or []
    mitigating = facts.get("mitigating") or []
    aggravating = facts.get("aggravating") or []
    punishments = facts.get("punishments") or []
    ml_role = payload.get("ml_role_model") or {}

    role_rows = [
        [
            item.get("label", ""),
            item.get("confidence", ""),
            _confidence_percent(item.get("confidence", "")),
            item.get("evidence", ""),
        ]
        for item in roles
    ]

    similar_rows: list[list[Any]] = []
    case_choices: list[tuple[str, str]] = []
    for item in payload.get("similar_cases") or []:
        score = f"{float(item.get('score', 0.0)):.1%}"
        case_number = item.get("case_number", "")
        similar_rows.append(
            [
                case_number,
                score,
                item.get("role_label", item.get("role", "")),
                item.get("punishment_label", item.get("punishment_type", "")),
                item.get("fragment", ""),
            ]
        )
        if case_number:
            case_choices.append((f"{case_number} · {score}", case_number))

    legal_rows = [
        [item.get("source", ""), f"{float(item.get('score', 0.0)):.1%}", item.get("text", "")]
        for item in payload.get("legal_sources") or []
    ]

    primary_role = roles[0]["label"] if roles else ml_role.get("predicted_role", "не определена")
    ml_role_line = f"ML-прогноз роли: {ml_role['predicted_role']}." if ml_role.get("available") and ml_role.get("predicted_role") else ""
    summary_lines = [
        f"Предварительная оценка основной роли: **{primary_role}**.",
        ml_role_line,
        f"Роли: {', '.join(f'{item.get('label', '')} ({_confidence_percent(item.get('confidence', ''))})' for item in roles) or 'не найдены'}.",
        f"Форма соучастия: {', '.join(item.get('label', '') for item in forms) or 'не найдена'}.",
        f"Нормы для проверки: {', '.join(articles) or 'не найдены'}.",
        f"Смягчающие обстоятельства: {', '.join(item.get('label', '') for item in mitigating) or 'не найдены'}.",
        f"Отягчающие обстоятельства: {', '.join(item.get('label', '') for item in aggravating) or 'не найдены'}.",
        f"Признаки наказания: {', '.join(item.get('label', '') for item in punishments) or 'не найдены'}.",
        f"Похожие дела: {len(similar_rows)}.",
        f"Правовые материалы: {len(legal_rows)}.",
        "",
        "Это аналитическая подсказка: окончательная квалификация и назначение наказания остаются за специалистом.",
    ]
    return "\n".join(line for line in summary_lines if line), role_rows, similar_rows, legal_rows, case_choices


def _chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": item["role"], "content": item["content"]}
        for item in messages
        if item.get("role") in {"user", "assistant"}
    ]


def restore_session(token: str) -> tuple[str, str, Any]:
    if not token:
        return "", "Создайте профиль или войдите по логину и паролю.", gr.update(choices=[], value=None)
    try:
        user = _request("GET", "/me", token)
        choices = _conversation_choices(token)
    except gr.Error:
        return "", "Сохранённый ключ больше не действует.", gr.update(choices=[], value=None)
    return token, f"Пользователь: **{user['display_name']}**", gr.update(choices=choices, value=None)


def register_user(display_name: str, username: str, password: str) -> tuple[str, str, Any]:
    if not display_name.strip():
        raise gr.Error("Введите имя пользователя.")
    if not username.strip():
        raise gr.Error("Введите логин.")
    if len(password) < 6:
        raise gr.Error("Пароль должен быть не короче 6 символов.")
    headers = {}
    if settings.registration_secret:
        headers["X-Registration-Secret"] = settings.registration_secret
    payload = _request(
        "POST",
        "/users",
        json={"display_name": display_name.strip(), "username": username.strip(), "password": password},
        headers=headers,
    )
    token = payload["api_token"]
    return token, f"Профиль **{payload['display_name']}** создан. Вход сохранён в этом браузере.", gr.update(choices=[], value=None)


def login_user(username: str, password: str) -> tuple[str, str, Any]:
    if not username.strip() or not password:
        raise gr.Error("Введите логин и пароль.")
    payload = _request("POST", "/login", json={"username": username.strip(), "password": password})
    token = payload["api_token"]
    choices = _conversation_choices(token)
    return token, f"Пользователь: **{payload['display_name']}**", gr.update(choices=choices, value=None)


def create_conversation(
    token: str,
    title: str,
    case_text: str,
    rag_profile: str,
) -> tuple[str, Any, list[Any], str, list[list[Any]], list[list[Any]], list[list[Any]], list[Any], Any, str, str]:
    if not token:
        raise gr.Error("Сначала войдите в профиль.")
    if not case_text.strip():
        raise gr.Error("Добавьте описание ситуации или текст судебного акта.")
    payload = _request(
        "POST",
        "/conversations",
        token,
        json={"title": title.strip() or None, "case_text": case_text, "rag_profile": rag_profile},
    )
    analysis = _request("POST", "/analyze", json={"text": case_text, "legal_top_k": 5, "case_top_k": 5})
    summary, role_rows, similar_rows, legal_rows, case_choices = _analysis_views(analysis)
    choices = _conversation_choices(token)
    return (
        payload["id"],
        gr.update(choices=choices, value=payload["id"]),
        [],
        summary,
        role_rows,
        similar_rows,
        legal_rows,
        [],
        gr.update(choices=case_choices, value=None),
        "",
        "Диалог создан. Задайте вопрос по материалам дела.",
    )


def open_conversation(
    token: str,
    conversation_id: str,
) -> tuple[str, str, str, str, list[dict[str, str]], str, list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]], Any, str, str]:
    if not token or not conversation_id:
        raise gr.Error("Выберите диалог.")
    payload = _request("GET", f"/conversations/{conversation_id}", token)
    assistant_messages = [item for item in payload["messages"] if item.get("role") == "assistant"]
    latest_sources = assistant_messages[-1].get("sources") if assistant_messages else []
    rows, case_choices = _sources_view(latest_sources)
    analysis = _request("POST", "/analyze", json={"text": payload["case_text"], "legal_top_k": 5, "case_top_k": 5})
    summary, role_rows, similar_rows, legal_rows, analysis_case_choices = _analysis_views(analysis)
    merged_case_choices = analysis_case_choices or case_choices
    return (
        payload["id"],
        payload["title"],
        payload["case_text"],
        payload["rag_profile"],
        _chat_messages(payload["messages"]),
        summary,
        role_rows,
        similar_rows,
        legal_rows,
        rows,
        gr.update(choices=merged_case_choices, value=None),
        "",
        f"Открыт диалог «{payload['title']}».",
    )


def new_conversation_form() -> tuple[str, Any, str, str, str, list[Any], str, list[Any], list[Any], list[Any], list[Any], Any, str, str]:
    return "", gr.update(value=None), "", "", "balanced", [], "", [], [], [], [], gr.update(choices=[], value=None), "", "Введите материалы нового дела."


def send_message(
    token: str,
    conversation_id: str,
    question: str,
    use_rag: bool,
    chat: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str, list[list[Any]], Any, str, Any]:
    if not token:
        raise gr.Error("Сначала войдите в профиль.")
    if not conversation_id:
        raise gr.Error("Создайте или откройте диалог.")
    if not question.strip():
        raise gr.Error("Введите вопрос.")
    payload = _request(
        "POST",
        f"/conversations/{conversation_id}/messages",
        token,
        json={"content": question.strip(), "use_rag": use_rag},
        timeout=1200,
    )
    updated_chat = list(chat or [])
    updated_chat.extend(
        [
            {"role": "user", "content": payload["user_message"]["content"]},
            {"role": "assistant", "content": payload["assistant_message"]["content"]},
        ]
    )
    sources = payload["assistant_message"].get("sources") or []
    rows, case_choices = _sources_view(sources)
    metrics = payload["assistant_message"].get("metrics") or {}
    if metrics.get("rag_enabled"):
        status = (
            f"RAG включён. Готово за {float(metrics.get('total_seconds', 0.0)):.1f} сек. "
            f"Поиск: {float(metrics.get('retrieval_seconds', 0.0)):.1f} сек., "
            f"ответ: {float(metrics.get('generation_seconds', 0.0)):.1f} сек. "
            f"Токенов ответа: {int(metrics.get('output_tokens', 0))}, "
            f"продолжений: {int(metrics.get('continuations_used', 0))}."
        )
    else:
        status = (
            f"RAG выключен. Готово за {float(metrics.get('total_seconds', 0.0)):.1f} сек. "
            f"Ответ: {float(metrics.get('generation_seconds', 0.0)):.1f} сек. "
            f"Токенов ответа: {int(metrics.get('output_tokens', 0))}, "
            f"продолжений: {int(metrics.get('continuations_used', 0))}."
        )
    return updated_chat, "", rows, gr.update(choices=case_choices, value=None), status, gr.update()


def open_full_case(case_number: str) -> str:
    if not case_number:
        raise gr.Error("Выберите судебное дело.")
    payload = _request("GET", "/case", params={"case_number": case_number})
    heading = " · ".join(
        value for value in [payload.get("court", ""), payload.get("date", ""), payload.get("article", "")] if value
    )
    return f"{heading}\n\n{payload.get('text', '')}".strip()


def delete_conversation(token: str, conversation_id: str) -> tuple[str, Any, list[Any], str, list[Any], list[Any], list[Any], list[Any], Any, str, str]:
    if not token or not conversation_id:
        raise gr.Error("Выберите диалог.")
    _request("DELETE", f"/conversations/{conversation_id}", token)
    choices = _conversation_choices(token)
    return "", gr.update(choices=choices, value=None), [], "", [], [], [], [], gr.update(choices=[], value=None), "", "Диалог удалён."


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="СППР") as demo:
        browser_token = gr.BrowserState(
            "",
            storage_key="sppr_api_token",
            secret=settings.ui_browser_secret,
        )
        current_conversation = gr.State("")

        gr.Markdown("# СППР по уголовным делам\nАнализ материалов, судебной практики и правовых оснований", elem_id="app-title")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                with gr.Accordion("Профиль", open=True):
                    profile_name = gr.Textbox(label="Имя", placeholder="Введите имя")
                    username = gr.Textbox(label="Логин", placeholder="Придумайте логин")
                    password = gr.Textbox(label="Пароль", type="password", placeholder="Придумайте пароль")
                    register_btn = gr.Button("Создать профиль", variant="primary")
                    login_btn = gr.Button("Войти")
                    profile_status = gr.Markdown("Создайте профиль или войдите по логину и паролю.")

                conversation_select = gr.Dropdown(label="Диалоги", choices=[], interactive=True)
                with gr.Row():
                    open_btn = gr.Button("Открыть", variant="primary")
                    new_btn = gr.Button("Новый")
                delete_btn = gr.Button("Удалить диалог", variant="stop")

                with gr.Accordion("Материалы дела", open=True):
                    case_title = gr.Textbox(label="Название", placeholder="Например: роль организатора")
                    case_text = gr.Textbox(
                        label="Описание ситуации или текст судебного акта",
                        lines=10,
                        max_lines=18,
                    )
                    rag_profile = gr.Dropdown(
                        choices=[("Быстрый", "fast"), ("Сбалансированный", "balanced"), ("Расширенный", "broad")],
                        value="balanced",
                        label="Глубина поиска",
                    )
                    create_btn = gr.Button("Создать диалог", variant="primary")

            with gr.Column(scale=3, min_width=600):
                chat = gr.Chatbot(
                    label="Диалог",
                    layout="panel",
                    height=760,
                    placeholder="Выберите диалог или создайте новый.",
                    elem_id="chat",
                )
                with gr.Row():
                    question = gr.Textbox(
                        label="Вопрос",
                        placeholder="Задайте вопрос по материалам дела",
                        lines=2,
                        scale=8,
                    )
                    send_btn = gr.Button("Отправить", variant="primary", scale=1)
                use_rag = gr.Checkbox(label="Использовать базу источников (RAG)", value=True)
                status = gr.Markdown("", elem_id="status")

                with gr.Tabs():
                    with gr.Tab("Сводка"):
                        analysis_summary = gr.Markdown("")
                    with gr.Tab("Роли"):
                        role_table = gr.Dataframe(
                            headers=["Роль", "Уверенность", "%", "Основание"],
                            datatype=["str", "str", "str", "str"],
                            value=[],
                            interactive=False,
                            wrap=True,
                            max_height=320,
                        )
                    with gr.Tab("Похожие дела"):
                        similar_cases = gr.Dataframe(
                            headers=["Дело", "Сходство", "Роль", "Наказание", "Фрагмент"],
                            datatype=["str", "str", "str", "str", "str"],
                            value=[],
                            interactive=False,
                            wrap=True,
                            max_height=360,
                        )
                    with gr.Tab("Правовые основания"):
                        legal_sources = gr.Dataframe(
                            headers=["Источник", "Сходство", "Фрагмент"],
                            datatype=["str", "str", "str"],
                            value=[],
                            interactive=False,
                            wrap=True,
                            max_height=360,
                        )
                    with gr.Tab("Источники ответа"):
                        sources = gr.Dataframe(
                            headers=["Ссылка", "Тип", "Источник", "Сведения", "Сходство"],
                            datatype=["str", "str", "str", "str", "str"],
                            value=[],
                            interactive=False,
                            wrap=True,
                            max_height=320,
                        )
                    with gr.Tab("Полный текст дела"):
                        case_source = gr.Dropdown(label="Судебное дело", choices=[])
                        open_case_btn = gr.Button("Открыть дело", variant="primary")
                        full_case = gr.Textbox(label="Текст судебного акта", lines=18, interactive=False)

        demo.load(
            restore_session,
            inputs=[browser_token],
            outputs=[browser_token, profile_status, conversation_select],
        )
        register_btn.click(
            register_user,
            inputs=[profile_name, username, password],
            outputs=[browser_token, profile_status, conversation_select],
        )
        login_btn.click(
            login_user,
            inputs=[username, password],
            outputs=[browser_token, profile_status, conversation_select],
        )
        new_btn.click(
            new_conversation_form,
            outputs=[
                current_conversation,
                conversation_select,
                case_title,
                case_text,
                rag_profile,
                chat,
                analysis_summary,
                role_table,
                similar_cases,
                legal_sources,
                sources,
                case_source,
                full_case,
                status,
            ],
        )
        create_btn.click(
            create_conversation,
            inputs=[browser_token, case_title, case_text, rag_profile],
            outputs=[
                current_conversation,
                conversation_select,
                chat,
                analysis_summary,
                role_table,
                similar_cases,
                legal_sources,
                sources,
                case_source,
                full_case,
                status,
            ],
        )
        open_btn.click(
            open_conversation,
            inputs=[browser_token, conversation_select],
            outputs=[
                current_conversation,
                case_title,
                case_text,
                rag_profile,
                chat,
                analysis_summary,
                role_table,
                similar_cases,
                legal_sources,
                sources,
                case_source,
                full_case,
                status,
            ],
        )
        send_btn.click(
            send_message,
            inputs=[browser_token, current_conversation, question, use_rag, chat],
            outputs=[chat, question, sources, case_source, status, conversation_select],
        )
        question.submit(
            send_message,
            inputs=[browser_token, current_conversation, question, use_rag, chat],
            outputs=[chat, question, sources, case_source, status, conversation_select],
        )
        delete_btn.click(
            delete_conversation,
            inputs=[browser_token, current_conversation],
            outputs=[
                current_conversation,
                conversation_select,
                chat,
                analysis_summary,
                role_table,
                similar_cases,
                legal_sources,
                sources,
                case_source,
                full_case,
                status,
            ],
        )
        open_case_btn.click(open_full_case, inputs=[case_source], outputs=[full_case])
    return demo
