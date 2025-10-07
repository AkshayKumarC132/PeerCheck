# rag_integration.py
from typing import Optional, Tuple
from peercheck import settings
from django.db import transaction
from .models import UserProfile, RAGAccount
from .ragitify_client import login, protected, register_user, vector_store_create, vector_store_list
import requests

def rag_on() -> bool:
    return bool(getattr(settings, "RAGITIFY_ENABLED", False))

def _base_url() -> str:
    return getattr(settings, "RAGITIFY_BASE_URL", "").rstrip("/")

def _timeout() -> int:
    return int(getattr(settings, "RAGITIFY_TIMEOUT_SECONDS", 30))

def _make_3pc_email(email_fallback: str, username: str) -> str:
    base_email = (email_fallback or f"{username}@example.com").strip().lower()
    if "@" in base_email:
        local, domain = base_email.split("@", 1)
        local = local.strip() or (username or "user")
        domain = domain.strip() or "example.com"
        return f"3pc_{local}@{domain}"
    return f"3pc_{(username or 'user').lower()}@example.com"

@transaction.atomic
def ensure_rag_token(user: UserProfile) -> Tuple[Optional[str], Optional[str]]:
    if not rag_on():
        return None, "RAG disabled"

    base_url = _base_url()
    if not base_url:
        return None, "RAG base URL not configured"

    default_password = getattr(settings, "RAGITIFY_DEFAULT_PASSWORD", "ChangeMe!123")
    rag_email = _make_3pc_email(user.email, user.username)

    acct, _created = RAGAccount.objects.get_or_create(user=user, defaults={"rag_email": rag_email})
    if acct.rag_email != rag_email:
        acct.rag_email = rag_email
        acct.save(update_fields=["rag_email"])

    if acct.access_token:
        try:
            protected(acct.access_token)
            return acct.access_token, None
        except Exception:
            pass

    try:
        print("RAG Register URL:", f"{base_url}/rag/register/")
        _ = register_user(
            username=rag_email,
            email=rag_email,
            password=default_password,
            tenant_name=f"3PC_{user.username}",
            collection_name=f"3PC_Collection-{user.username}",
        )
    except requests.HTTPError as e:
        try:
            print("RAG register error:", e.response.status_code, e.response.text)
        except Exception:
            print("RAG register error:", e)
    except Exception as e:
        print("RAG register request exception:", e)

    try:
        print("trying RAG login for", acct.rag_email)
        resp = login(email=acct.rag_email, password=default_password)
        print("RAG login response:", resp)
        token = (resp or {}).get("token") or (resp or {}).get("access_token")
        if not token:
            return None, "RAG login failed (no token in response)"
        acct.access_token = token
        acct.save(update_fields=["access_token"])
        return token, None
    except Exception as e:
        return None, f"RAG login error: {e}"

@transaction.atomic
def ensure_user_vector_store_id(user: UserProfile) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Ensure a SINGLE Vector Store exists for this 3PC user.
    Returns (token, vector_store_id, error_message)
    - Reuses RAGAccount.vector_store_id when present.
    - Otherwise tries to find an existing VS by canonical name.
    - Otherwise creates once and stores in RAGAccount.
    """
    token, err = ensure_rag_token(user)
    if not token:
        return None, None, err or "Unable to get RAG token"

    acct: RAGAccount = RAGAccount.objects.get(user=user)

    # If user already has a VS, reuse it
    if acct.vector_store_id:
        return token, acct.vector_store_id, None

    # Canonical VS name for this user
    canonical_name = f"3PC-{user.username}"

    # Try to discover an existing VS from RAG by name
    try:
        vs_list = vector_store_list(token) or []
        match = None
        for item in vs_list:
            # RAG can return different keys; handle defensively
            name = item.get("name") or item.get("title") or ""
            _id = item.get("id") or item.get("uuid") or item.get("vector_store_id")
            if not _id:
                continue
            if name.strip().lower() == canonical_name.strip().lower():
                match = str(_id)
                break
        if match:
            acct.vector_store_id = match
            acct.save(update_fields=["vector_store_id"])
            return token, match, None
    except Exception as e:
        # non-fatal; fall through to create
        print("vector_store_list error:", e)

    # Create ONCE if not found
    try:
        created = vector_store_create(token, name=canonical_name) or {}
        vs_id = created.get("id") or created.get("uuid") or created.get("vector_store_id")
        if not vs_id:
            return token, None, "Vector store create returned no id"
        acct.vector_store_id = str(vs_id)
        acct.save(update_fields=["vector_store_id"])
        return token, acct.vector_store_id, None
    except Exception as e:
        return token, None, f"Vector store create error: {e}"
