# ragitify_client.py
import requests
from typing import Any, Dict, List, Optional
from peercheck import settings

def _enabled() -> bool:
    return bool(getattr(settings, "RAGITIFY_ENABLED", False)) and bool(getattr(settings, "RAGITIFY_BASE_URL", ""))

def _path(key: str, **fmt) -> str:
    """
    Paths are provided via settings.RAGITIFY_PATHS and must include all keys used here:
      register, login, protected,
      vector_store_create, vector_store_list, vector_store_detail,
      document_ingest, document_list, document_detail, document_status,
      assistant_create, assistant_list, assistant_detail,
      thread_create, thread_list, thread_detail, thread_messages,
      message_create, message_list, message_detail,
      run_create, run_list, run_detail, run_submit_tool_outputs
    """
    base = getattr(settings, "RAGITIFY_BASE_URL", "").rstrip("/")
    tpl = settings.RAGITIFY_PATHS.get(key, "")
    if not tpl:
        raise ValueError(f"Missing RAG path '{key}'")
    return base + tpl.format(**fmt)

def _headers() -> Dict[str, str]:
    return {"Content-Type": "application/json"}

def _timeout() -> int:
    return int(getattr(settings, "RAGITIFY_TIMEOUT_SECONDS", 30))

def _req(method: str, url: str, json: Optional[dict] = None):
    resp = requests.request(method, url, headers=_headers(), json=json, timeout=_timeout())
    resp.raise_for_status()
    if resp.headers.get("Content-Type", "").startswith("application/json"):
        return resp.json()
    # Some endpoints (e.g., 204) may not return JSON.
    try:
        return resp.json()
    except Exception:
        return {}

# ---------------- AUTH ----------------

def register_user(*, username: str, email: str, password: str, tenant_name: str, collection_name: str):
    """
    RAG register expects: username, email, password, tenant_name, collection_name
    """
    if not _enabled(): return {}
    url = _path("register")
    payload = {
        "username": username,
        "email": email,
        "password": password,
        "tenant_name": tenant_name,
        "collection_name": collection_name,
    }
    return _req("POST", url, json=payload)

def login(*, email: str, password: str):
    if not _enabled(): return {}
    url = _path("login")
    return _req("POST", url, json={"email": email, "password": password})

def protected(token: str):
    if not _enabled(): return {}
    url = _path("protected", token=token)
    return _req("GET", url)

# ---------------- VECTOR STORE ----------------

def vector_store_create(token: str, *, name: str):
    if not _enabled(): return {}
    url = _path("vector_store_create", token=token)
    return _req("POST", url, json={"name": name})

def vector_store_list(token: str):
    if not _enabled(): return []
    url = _path("vector_store_list", token=token)
    data = _req("GET", url)
    return data if isinstance(data, list) else data.get("results", data) or []

def vector_store_detail(token: str, store_id: str):
    if not _enabled(): return {}
    url = _path("vector_store_detail", token=token, id=store_id)
    return _req("GET", url)

# ---------------- DOCUMENTS ----------------

# ragitify_client.py (only this function changed)
def document_ingest(
    token: str,
    *,
    vector_store_id: str,
    s3_file_url: Optional[str] = None,
    file: Optional[object] = None,
):
    """
    Ingest a document into RAGitify.

    When 'file' is provided (e.g., Django InMemoryUploadedFile), send multipart/form-data:
      - data: { vector_store_id, [s3_file_url] }
      - files: { file: (filename, fileobj, content_type) }

    Otherwise, send JSON with { vector_store_id, s3_file_url }.
    """
    if not _enabled():
        return {}

    url = _path("document_ingest", token=token)

    if not (file or s3_file_url):
        raise ValueError("Provide either 'file' (multipart) or 's3_file_url' (JSON).")

    # ---- Multipart form-data branch (file upload) ----
    if file is not None:
        data = {"vector_store_id": vector_store_id}
        if s3_file_url:
            # RAG accepts both file and s3_file_url; include if you want RAG to fetch too.
            data["s3_file_url"] = s3_file_url

        filename = getattr(file, "name", "upload.dat")
        content_type = getattr(file, "content_type", "application/octet-stream")
        files = {"file": (filename, file, content_type)}

        resp = requests.post(url, data=data, files=files, timeout=_timeout())
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {}

    # ---- JSON branch (S3 URL ingestion) ----
    payload: Dict[str, Any] = {"vector_store_id": vector_store_id, "s3_file_url": s3_file_url}
    print("Document ingest payload:", payload)
    return _req("POST", url, json=payload)

def document_status(token: str, *, document_id: str):
    if not _enabled(): return {}
    url = _path("document_status", token=token, document_id=document_id)
    return _req("GET", url)

def document_list(token: str):
    if not _enabled(): return []
    url = _path("document_list", token=token)
    data = _req("GET", url)
    return data if isinstance(data, list) else data.get("results", data) or []

def document_detail(token: str, doc_id: str):
    if not _enabled(): return {}
    url = _path("document_detail", token=token, id=doc_id)
    return _req("GET", url)

# ---------------- ASSISTANT ----------------
# RAG AssistantSerializer: name, vector_store_id, instructions?, model?, tools? (list)
# (no vector_store_ids list)
def assistant_create(token: str, *, name: str, vector_store_id: str, instructions: str = "", model: str = "", tools: Optional[List[Dict[str, Any]]] = None):
    if not _enabled(): return {}
    url = _path("assistant_create", token=token)
    payload: Dict[str, Any] = {"name": name, "vector_store_id": vector_store_id}
    if instructions:
        payload["instructions"] = instructions
    if model:
        payload["model"] = model
    if tools is not None:
        payload["tools"] = tools
    return _req("POST", url, json=payload)

def assistant_list(token: str):
    if not _enabled(): return []
    url = _path("assistant_list", token=token)
    data = _req("GET", url)
    return data if isinstance(data, list) else data.get("results", data) or []

def assistant_detail(token: str, assistant_id: str):
    if not _enabled(): return {}
    url = _path("assistant_detail", token=token, id=assistant_id)
    return _req("GET", url)

# ---------------- THREAD ----------------
# RAG ThreadSerializer: vector_store_id (required), optional title
def thread_create(token: str, *, vector_store_id: str, title: Optional[str] = None):
    if not _enabled(): return {}
    url = _path("thread_create", token=token)
    payload: Dict[str, Any] = {"vector_store_id": vector_store_id}
    if title:
        payload["title"] = title
    return _req("POST", url, json=payload)

def thread_list(token: str):
    if not _enabled(): return []
    url = _path("thread_list", token=token)
    data = _req("GET", url)
    return data if isinstance(data, list) else data.get("results", data) or []

def thread_detail(token: str, thread_id: str):
    if not _enabled(): return {}
    url = _path("thread_detail", token=token, id=thread_id)
    return _req("GET", url)

def thread_messages(token: str, thread_id: str):
    if not _enabled(): return []
    url = _path("thread_messages", token=token, thread_id=thread_id)
    data = _req("GET", url)
    return data if isinstance(data, list) else data.get("results", data) or []

# ---------------- MESSAGE ----------------
# RAG MessageSerializer: thread_id (required), content (required); role defaults to "user"
def message_create(token: str, *, thread_id: str, content: str, role: str = "user"):
    if not _enabled(): return {}
    url = _path("message_create", token=token)
    payload = {"thread_id": thread_id, "content": content}
    if role:
        payload["role"] = role
    return _req("POST", url, json=payload)

def message_list(token: str):
    if not _enabled(): return []
    url = _path("message_list", token=token)
    data = _req("GET", url)
    return data if isinstance(data, list) else data.get("results", data) or []

def message_detail(token: str, message_id: str):
    if not _enabled(): return {}
    url = _path("message_detail", token=token, id=message_id)
    return _req("GET", url)

# ---------------- RUN ----------------
# RAG RunSerializer: thread_id, assistant_id, optional mode, optional message_id/source_run_id; submit-tool-outputs requires {tool_outputs:[{tool_call_id, output}]}
def run_create(token: str, *, thread_id: str, assistant_id: str, mode: Optional[str] = None, message_id: Optional[int] = None, source_run_id: Optional[str] = None):
    if not _enabled(): return {}
    url = _path("run_create", token=token)
    payload: Dict[str, Any] = {"thread_id": thread_id, "assistant_id": assistant_id}
    if mode:
        payload["mode"] = mode
    if message_id is not None:
        payload["message_id"] = message_id
    if source_run_id:
        payload["source_run_id"] = source_run_id
    return _req("POST", url, json=payload)

def run_list(token: str):
    if not _enabled(): return []
    url = _path("run_list", token=token)
    data = _req("GET", url)
    return data if isinstance(data, list) else data.get("results", data) or []

def run_detail(token: str, run_id: str):
    if not _enabled(): return {}
    url = _path("run_detail", token=token, id=run_id)
    return _req("GET", url)

def run_submit_tool_outputs(token: str, *, run_id: str, tool_outputs: List[Dict[str, Any]]):
    if not _enabled(): return {}
    url = _path("run_submit_tool_outputs", token=token, run_id=run_id)
    payload = {"tool_outputs": tool_outputs}
    return _req("POST", url, json=payload)
