from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .config import settings


RagProfileName = Literal["fast", "balanced", "broad"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=12000)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    legal_top_k: int = 5
    case_top_k: int = 5


class SimilarCasesRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 5


class ChatContextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    rag_profile: RagProfileName = settings.rag_profile
    legal_top_k: int | None = Field(default=None, ge=1, le=20)
    case_top_k: int | None = Field(default=None, ge=1, le=20)


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list)
    use_rag: bool = True
    rag_profile: RagProfileName = settings.rag_profile
    legal_top_k: int | None = Field(default=None, ge=1, le=20)
    case_top_k: int | None = Field(default=None, ge=1, le=20)
    return_context: bool = False


class UserCreate(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=120)
    username: str = Field(..., min_length=3, max_length=120)
    password: str = Field(..., min_length=6, max_length=200)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=120)
    password: str = Field(..., min_length=1, max_length=200)


class ConversationCreate(BaseModel):
    case_text: str = Field(..., min_length=1, max_length=200_000)
    title: str | None = Field(default=None, max_length=200)
    rag_profile: RagProfileName = settings.rag_profile


class ConversationMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=8000)
    use_rag: bool = True
    rag_profile: RagProfileName | None = None
