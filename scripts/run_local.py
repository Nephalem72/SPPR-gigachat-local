from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def load_env() -> None:
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_data_files() -> None:
    data_dir = Path(os.getenv("SPPR_DATA_DIR", "D:/Notebooks/sppr"))
    required = [
        "laws.parquet",
        "final_roles_punishments_v3.parquet",
        "cases_with_id.parquet",
        "role_model.pkl",
        "vectorizer.pkl",
        "embeddings.pkl",
        "faiss_index.bin",
    ]
    missing = [str(data_dir / name) for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Не найдены файлы данных:\n" + "\n".join(missing))


def wait_fastapi(url: str, process: subprocess.Popen[str]) -> None:
    for second in range(1, 121):
        if process.poll() is not None:
            raise RuntimeError(f"FastAPI завершился с кодом {process.returncode}")
        try:
            response = requests.get(f"{url}/health", timeout=3)
            if response.ok:
                print("FastAPI готов:", response.json(), flush=True)
                return
        except requests.RequestException:
            pass
        if second % 10 == 0:
            print(f"Жду FastAPI: {second} сек.", flush=True)
        time.sleep(1)
    raise TimeoutError("FastAPI не поднялся за 120 секунд")


def find_free_port(preferred: int, host: str = "127.0.0.1", attempts: int = 50) -> int:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
    raise OSError(f"Не найден свободный порт в диапазоне {preferred}-{preferred + attempts - 1}")


def main() -> None:
    load_env()
    os.environ.setdefault("SPPR_LLM_BACKEND", "gigachat")
    os.environ.setdefault("SPPR_LLM_MODEL_ID", "GigaChat-2")
    os.environ.setdefault("SPPR_DATA_DIR", "D:/Notebooks/sppr")
    os.environ.setdefault("SPPR_DATABASE_URL", "sqlite:///D:/Notebooks/sppr/sppr_history.db")
    os.environ.setdefault("SPPR_HOST", "127.0.0.1")
    os.environ.setdefault("SPPR_UI_HOST", "127.0.0.1")
    os.environ.setdefault("SPPR_UI_SHARE", "false")

    if os.environ.get("SPPR_LLM_BACKEND") == "gigachat" and not os.environ.get("SPPR_GIGACHAT_AUTH_DATA"):
        raise RuntimeError("Заполните SPPR_GIGACHAT_AUTH_DATA в .env")

    require_data_files()
    os.chdir(ROOT)

    api_url = os.environ.get("SPPR_API_URL", "http://127.0.0.1:8000")
    api_process = subprocess.Popen([sys.executable, "app_fastapi.py"], text=True)
    try:
        wait_fastapi(api_url, api_process)
        ui_host = os.environ.get("SPPR_UI_HOST", "127.0.0.1")
        preferred_ui_port = int(os.environ.get("SPPR_UI_PORT", "7860"))
        ui_port = find_free_port(preferred_ui_port, ui_host)
        os.environ["SPPR_UI_PORT"] = str(ui_port)
        print(f"Запускаю Gradio UI: http://{ui_host}:{ui_port}", flush=True)
        subprocess.run([sys.executable, "app_gradio.py"], check=True)
    finally:
        if api_process.poll() is None:
            api_process.terminate()


if __name__ == "__main__":
    main()
