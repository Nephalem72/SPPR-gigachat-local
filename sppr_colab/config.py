from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    drive_root: Path = Path(os.getenv("SPPR_DRIVE_ROOT", "D:/Notebooks/sppr"))
    data_dir: Path = Path(os.getenv("SPPR_DATA_DIR", "D:/Notebooks/sppr"))
    host: str = os.getenv("SPPR_HOST", "0.0.0.0")
    port: int = int(os.getenv("SPPR_PORT", "8000"))
    legacy_case_encoder: str = os.getenv(
        "SPPR_LEGACY_CASE_ENCODER",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    llm_model_id: str = os.getenv("SPPR_LLM_MODEL_ID", "GigaChat-2")
    llm_backend: str = os.getenv("SPPR_LLM_BACKEND", "gigachat")
    llm_max_new_tokens: int = int(os.getenv("SPPR_LLM_MAX_NEW_TOKENS", "768"))
    llm_max_continuations: int = int(os.getenv("SPPR_LLM_MAX_CONTINUATIONS", "2"))
    llm_max_input_tokens: int = int(os.getenv("SPPR_LLM_MAX_INPUT_TOKENS", "6144"))
    llm_temperature: float = float(os.getenv("SPPR_LLM_TEMPERATURE", "0.0"))
    llm_top_p: float = float(os.getenv("SPPR_LLM_TOP_P", "0.9"))
    llm_load_in_4bit: bool = _env_bool("SPPR_LLM_LOAD_IN_4BIT", False)
    gigachat_auth_data: str = os.getenv("SPPR_GIGACHAT_AUTH_DATA", "")
    gigachat_scope: str = os.getenv("SPPR_GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    gigachat_oauth_url: str = os.getenv("SPPR_GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
    gigachat_api_url: str = os.getenv("SPPR_GIGACHAT_API_URL", "https://gigachat.devices.sberbank.ru/api/v1")
    gigachat_verify_ssl: bool = _env_bool("SPPR_GIGACHAT_VERIFY_SSL", True)
    gigachat_timeout: int = int(os.getenv("SPPR_GIGACHAT_TIMEOUT", "120"))
    rag_profile: str = os.getenv("SPPR_RAG_PROFILE", "balanced")
    max_history_messages: int = int(os.getenv("SPPR_MAX_HISTORY_MESSAGES", "8"))
    database_url: str = os.getenv("SPPR_DATABASE_URL", "")
    allow_user_registration: bool = _env_bool("SPPR_ALLOW_USER_REGISTRATION", True)
    registration_secret: str = os.getenv("SPPR_REGISTRATION_SECRET", "")
    api_url: str = os.getenv("SPPR_API_URL", "http://127.0.0.1:8000")
    ui_host: str = os.getenv("SPPR_UI_HOST", "0.0.0.0")
    ui_port: int = int(os.getenv("SPPR_UI_PORT", "7860"))
    ui_share: bool = _env_bool("SPPR_UI_SHARE", False)
    ui_username: str = os.getenv("SPPR_UI_USERNAME", "")
    ui_password: str = os.getenv("SPPR_UI_PASSWORD", "")
    ui_browser_secret: str = os.getenv("SPPR_UI_BROWSER_SECRET", "sppr-colab-ui-v1")

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'sppr_history.db').as_posix()}"

    @property
    def laws_path(self) -> Path:
        return self.data_dir / "laws.parquet"

    @property
    def final_dataset_path(self) -> Path:
        return self.data_dir / "final_roles_punishments_v3.parquet"

    @property
    def cases_path(self) -> Path:
        return self.data_dir / "cases_with_id.parquet"

    @property
    def role_model_path(self) -> Path:
        return self.data_dir / "role_model.pkl"

    @property
    def role_vectorizer_path(self) -> Path:
        return self.data_dir / "vectorizer.pkl"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings.pkl"

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "faiss_index.bin"


settings = Settings()
