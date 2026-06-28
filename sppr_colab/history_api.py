from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from .auth import AuthenticatedUser, create_api_token, get_current_user, hash_password, verify_password
from .config import settings
from .database import Conversation, Message, User, get_db, utc_now
from .schemas import ConversationCreate, ConversationMessageRequest, LoginRequest, UserCreate
from .service import answer_chat


router = APIRouter(tags=["users and conversations"])


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _get_owned_conversation(db: Session, conversation_id: str, user_id: str) -> Conversation:
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    return conversation


def _message_payload(message: Message) -> dict[str, Any]:
    return {
        "id": message.id,
        "sequence": message.sequence,
        "role": message.role,
        "content": message.content,
        "model_id": message.model_id,
        "sources": message.sources,
        "metrics": message.metrics,
        "created_at": _iso(message.created_at),
    }


@router.post("/users", status_code=status.HTTP_201_CREATED)
def register_user(
    request: UserCreate,
    registration_secret: str | None = Header(default=None, alias="X-Registration-Secret"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not settings.allow_user_registration:
        raise HTTPException(status_code=403, detail="Регистрация отключена")
    if settings.registration_secret and not secrets.compare_digest(
        registration_secret or "",
        settings.registration_secret,
    ):
        raise HTTPException(status_code=403, detail="Недействительный ключ регистрации")
    username = request.username.strip().lower()
    if db.scalar(select(User).where(User.username == username)) is not None:
        raise HTTPException(status_code=409, detail="Логин уже занят")
    api_token, token_hash = create_api_token()
    user = User(
        display_name=request.display_name.strip(),
        username=username,
        password_hash=hash_password(request.password),
        token_hash=token_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "id": user.id,
        "display_name": user.display_name,
        "api_token": api_token,
        "created_at": _iso(user.created_at),
        "token_notice": "Сохраните токен: повторно он не показывается.",
    }


@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    username = request.username.strip().lower()
    user = db.scalar(select(User).where(User.username == username))
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    api_token, token_hash = create_api_token()
    user.token_hash = token_hash
    db.commit()
    return {
        "id": user.id,
        "display_name": user.display_name,
        "api_token": api_token,
        "created_at": _iso(user.created_at),
    }


@router.get("/me")
def current_user(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, str]:
    return {"id": user.id, "display_name": user.display_name}


@router.post("/me/token")
def rotate_token(
    user: AuthenticatedUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    stored_user = db.get(User, user.id)
    if stored_user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    api_token, token_hash = create_api_token()
    stored_user.token_hash = token_hash
    db.commit()
    return {"api_token": api_token, "token_notice": "Предыдущий токен больше не действует."}


@router.delete("/me")
def delete_user(
    user: AuthenticatedUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, bool]:
    stored_user = db.get(User, user.id)
    if stored_user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    db.delete(stored_user)
    db.commit()
    return {"deleted": True}


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: ConversationCreate,
    user: AuthenticatedUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    title = request.title.strip() if request.title else request.case_text.strip().replace("\n", " ")[:80]
    conversation = Conversation(
        user_id=user.id,
        title=title or "Новый диалог",
        case_text=request.case_text,
        rag_profile=request.rag_profile,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return {
        "id": conversation.id,
        "title": conversation.title,
        "rag_profile": conversation.rag_profile,
        "created_at": _iso(conversation.created_at),
    }


@router.get("/conversations")
def list_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: AuthenticatedUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    total = db.scalar(select(func.count()).select_from(Conversation).where(Conversation.user_id == user.id)) or 0
    items = db.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(desc(Conversation.updated_at))
        .offset(offset)
        .limit(limit)
    ).all()
    return {
        "total": total,
        "items": [
            {
                "id": item.id,
                "title": item.title,
                "rag_profile": item.rag_profile,
                "created_at": _iso(item.created_at),
                "updated_at": _iso(item.updated_at),
            }
            for item in items
        ],
    }


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    message_limit: int = Query(default=100, ge=1, le=500),
    user: AuthenticatedUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    conversation = _get_owned_conversation(db, conversation_id, user.id)
    messages = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(desc(Message.sequence))
        .limit(message_limit)
    ).all()
    return {
        "id": conversation.id,
        "title": conversation.title,
        "case_text": conversation.case_text,
        "rag_profile": conversation.rag_profile,
        "created_at": _iso(conversation.created_at),
        "updated_at": _iso(conversation.updated_at),
        "messages": [_message_payload(item) for item in reversed(messages)],
    }


@router.post("/conversations/{conversation_id}/messages")
def send_message(
    conversation_id: str,
    request: ConversationMessageRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    conversation = _get_owned_conversation(db, conversation_id, user.id)
    previous = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(desc(Message.sequence))
        .limit(settings.max_history_messages)
    ).all()
    history = [
        {"role": item.role, "content": item.content}
        for item in reversed(previous)
        if item.role in {"user", "assistant"}
    ]
    case_text = conversation.case_text
    rag_profile = request.rag_profile or conversation.rag_profile
    db.commit()

    response = answer_chat(
        case_text,
        request.content,
        history=history,
        use_rag=request.use_rag,
        rag_profile=rag_profile,
        return_context=False,
    )

    conversation = db.scalar(
        select(Conversation).where(Conversation.id == conversation.id).with_for_update()
    )
    last_sequence = db.scalar(
        select(func.max(Message.sequence)).where(Message.conversation_id == conversation.id)
    ) or 0
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        sequence=last_sequence + 1,
        content=request.content,
    )
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        sequence=last_sequence + 2,
        content=response["answer"],
        model_id=response["model"]["model_id"],
        sources=response["sources"],
        metrics=response["metrics"],
    )
    conversation.rag_profile = rag_profile
    conversation.updated_at = utc_now()
    db.add_all([user_message, assistant_message])
    db.commit()
    db.refresh(user_message)
    db.refresh(assistant_message)
    return {
        "user_message": _message_payload(user_message),
        "assistant_message": _message_payload(assistant_message),
        "citation_check": response["citation_check"],
    }


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, bool]:
    conversation = _get_owned_conversation(db, conversation_id, user.id)
    db.delete(conversation)
    db.commit()
    return {"deleted": True}
