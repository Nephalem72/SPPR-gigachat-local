from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schemas import AnalyzeRequest, ChatContextRequest, ChatRequest, SimilarCasesRequest
from .config import settings
from .llm import warmup_llm
from .rag import RAG_PROFILES, profile_payload
from .service import analyze_text, answer_chat, build_chat_context, health_status, warmup
from .retrieval import fetch_full_cases, search_similar_cases
from .database import database_health, init_db
from .history_api import router as history_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="SPPR Colab Backend", lifespan=lifespan)
app.include_router(history_router)


@app.get("/health")
def health() -> dict:
    return {**health_status(), "database": database_health()}


@app.get("/warmup")
def warmup_endpoint(include_llm: bool = False) -> dict:
    result = {"retrieval": warmup()}
    if include_llm:
        result["llm"] = warmup_llm()
    return result


@app.get("/config")
def config() -> dict:
    return {
        "llm_backend": settings.llm_backend,
        "llm_model_id": settings.llm_model_id,
        "llm_load_in_4bit": settings.llm_load_in_4bit,
        "gigachat_configured": bool(settings.gigachat_auth_data),
        "gigachat_scope": settings.gigachat_scope if settings.llm_backend == "gigachat" else None,
        "rag_profile": settings.rag_profile,
        "rag_profiles": {name: profile_payload(profile) for name, profile in RAG_PROFILES.items()},
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    return analyze_text(request.text, legal_top_k=request.legal_top_k, case_top_k=request.case_top_k)


@app.post("/similar_cases")
def similar_cases(request: SimilarCasesRequest) -> dict:
    return {"query": request.query, "items": search_similar_cases(request.query, request.top_k)}


@app.get("/case")
def full_case(case_number: str) -> dict:
    item = fetch_full_cases((case_number,)).get(case_number)
    if item is None:
        raise HTTPException(status_code=404, detail="Дело не найдено")
    return item


@app.post("/chat_context")
def chat_context(request: ChatContextRequest) -> dict:
    return build_chat_context(
        request.text,
        rag_profile=request.rag_profile,
        legal_top_k=request.legal_top_k,
        case_top_k=request.case_top_k,
    )


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    return answer_chat(
        request.text,
        request.question,
        history=[item.model_dump() for item in request.history],
        use_rag=request.use_rag,
        rag_profile=request.rag_profile,
        legal_top_k=request.legal_top_k,
        case_top_k=request.case_top_k,
        return_context=request.return_context,
    )
