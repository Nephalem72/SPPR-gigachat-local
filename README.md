# SPPR GigaChat Local

Локальный сервер СППР: данные, поиск, RAG и история работают на компьютере, а генерация ответа уходит во внешний GigaChat API.

## Что остается локально

- `laws.parquet`
- `final_roles_punishments_v3.parquet`
- `cases_with_id.parquet`
- `role_model.pkl`
- `vectorizer.pkl`
- `embeddings.pkl`
- `faiss_index.bin`
- SQLite-база истории пользователей и диалогов
- FastAPI + Gradio UI

По умолчанию данные берутся из:

```text
D:/Notebooks/sppr
```

## Настройка

Скопируйте `.env.example` в `.env` и заполните GigaChat authorization key:

```powershell
copy .env.example .env
```

Минимально нужно задать:

```text
SPPR_GIGACHAT_AUTH_DATA=...
```

`SPPR_GIGACHAT_AUTH_DATA` — это base64 authorization data из кабинета GigaChat API, без префикса `Basic`.

Если на Windows возникнет ошибка сертификата, для локального теста можно временно поставить:

```text
SPPR_GIGACHAT_VERIFY_SSL=false
```

## Запуск

```powershell
cd D:\GitHub\SPPR-gigachat-local
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts\run_local.py
```

После запуска:

- FastAPI: `http://127.0.0.1:8000`
- Gradio UI: `http://127.0.0.1:7860`

## Основные переменные окружения

- `SPPR_DATA_DIR` — папка с parquet/pkl/faiss, по умолчанию `D:/Notebooks/sppr`
- `SPPR_LLM_BACKEND` — `gigachat` или `transformers`, по умолчанию `gigachat`
- `SPPR_LLM_MODEL_ID` — модель GigaChat, по умолчанию `GigaChat`
- `SPPR_GIGACHAT_AUTH_DATA` — обязательный ключ для GigaChat API
- `SPPR_GIGACHAT_SCOPE` — scope OAuth, по умолчанию `GIGACHAT_API_PERS`
- `SPPR_GIGACHAT_VERIFY_SSL` — проверка SSL, по умолчанию `true`
- `SPPR_LLM_MAX_NEW_TOKENS` — лимит ответа, по умолчанию `768`
- `SPPR_RAG_PROFILE` — `fast`, `balanced`, `broad`
- `SPPR_DATABASE_URL` — БД истории, по умолчанию SQLite в `D:/Notebooks/sppr/sppr_history.db`

## Проверка API

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/config
```

## Архитектура

```text
Пользователь
  -> Gradio UI
  -> FastAPI
  -> локальный RAG: УК РФ + судебные дела + признаки
  -> GigaChat API
  -> ответ + источники + история
```
