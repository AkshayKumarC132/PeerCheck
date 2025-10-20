"""Utilities for orchestrating RAGitify document matching for audio transcripts."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from django.db import close_old_connections
from django.utils import timezone

from peercheck import settings

from .models import AudioFile, ProcessingSession, RAGAccount, ReferenceDocument, UserProfile
from .rag_integration import ensure_user_vector_store_id
from .ragitify_client import (
    assistant_create,
    run_create,
    run_detail,
    thread_create,
    thread_messages,
    message_create,
)

logger = logging.getLogger(__name__)


MATCH_ASSISTANT_NAME_TEMPLATE = "3PC-DocMatcher-{username}"
MATCH_ASSISTANT_INSTRUCTIONS = (
    "You are a helpful assistant that receives a surgical transcription and must identify the most relevant "
    "reference documents that the transcript likely corresponds to. Respond strictly in JSON using the following "
    "schema:\n"
    "{\n"
    "  \"documents\": [\n"
    "    {\n"
    "      \"rag_document_id\": \"string\",\n"
    "      \"reference_document_name\": \"string\",\n"
    "      \"confidence\": 0.0\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "List up to five documents ordered by confidence (0.0-1.0). If no documents are relevant, return an empty array."
)

MAX_TRANSCRIPT_CHARS = 6000
RUN_POLL_INTERVAL_SECONDS = 2
RUN_POLL_ATTEMPTS = 10
SIMILARITY_DELTA = getattr(settings, "RAG_DOCUMENT_SIMILARITY_DELTA", 0.05)
MIN_CONFIDENCE = getattr(settings, "RAG_DOCUMENT_MIN_CONFIDENCE", 0.55)


def rag_feature_enabled() -> bool:
    return bool(getattr(settings, "RAGITIFY_ENABLED", False)) and bool(getattr(settings, "RAGITIFY_BASE_URL", ""))


@dataclass
class MatchResult:
    status: str
    documents: List[Dict[str, Any]]
    selected_reference_id: Optional[str] = None
    error: Optional[str] = None


def _ensure_assistant(user: UserProfile) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Ensure a document matching assistant exists for the user."""
    if not rag_feature_enabled():
        return None, None, "RAG disabled"

    token, vector_store_id, err = ensure_user_vector_store_id(user)
    if not token or not vector_store_id:
        return token, None, err or "Missing vector store"

    account: RAGAccount = RAGAccount.objects.get(user=user)
    if account.document_match_assistant_id:
        return token, account.document_match_assistant_id, None

    try:
        name = MATCH_ASSISTANT_NAME_TEMPLATE.format(username=user.username)
        response = assistant_create(
            token,
            name=name,
            vector_store_id=vector_store_id,
            instructions=MATCH_ASSISTANT_INSTRUCTIONS,
        )
        assistant_id = (response or {}).get("id") or (response or {}).get("assistant_id")
        if not assistant_id:
            return token, None, "Assistant create returned no identifier"
        account.document_match_assistant_id = str(assistant_id)
        account.save(update_fields=["document_match_assistant_id"])
        return token, account.document_match_assistant_id, None
    except Exception as exc:  # pragma: no cover - network failure path
        logger.exception("Failed to create document matching assistant for user %s", user.id)
        return token, None, str(exc)


def schedule_document_match(audio_file: AudioFile) -> Optional[threading.Thread]:
    """Kick off asynchronous document matching for a processed audio file."""
    if not rag_feature_enabled():
        return None
    if not audio_file or not audio_file.user:
        return None
    transcript_text = (audio_file.transcription or {}).get("text") if audio_file.transcription else None
    if not transcript_text:
        return None

    # mark as pending immediately to inform callers
    audio_file.rag_document_match_status = "pending"
    audio_file.rag_document_match_error = None
    audio_file.rag_document_match_updated_at = timezone.now()
    audio_file.rag_document_matches = {
        "documents": [],
        "selected_reference_document_id": None,
    }
    audio_file.save(
        update_fields=[
            "rag_document_match_status",
            "rag_document_match_error",
            "rag_document_match_updated_at",
            "rag_document_matches",
        ]
    )

    thread = threading.Thread(target=_run_document_match, args=(audio_file.id,), daemon=True)
    thread.start()
    return thread


def _run_document_match(audio_file_id):
    close_old_connections()
    try:
        audio = AudioFile.objects.select_related("user").get(id=audio_file_id)
    except AudioFile.DoesNotExist:  # pragma: no cover - defensive
        return

    user = audio.user
    if not user:
        _persist_result(audio, MatchResult(status="error", documents=[], error="Audio file has no owner"))
        return

    transcript_text = (audio.transcription or {}).get("text") if audio.transcription else None
    if not transcript_text:
        _persist_result(audio, MatchResult(status="error", documents=[], error="Transcript missing for audio"))
        return

    token, assistant_id, err = _ensure_assistant(user)
    if not assistant_id or not token:
        _persist_result(audio, MatchResult(status="error", documents=[], error=err or "Assistant unavailable"))
        return

    account = RAGAccount.objects.get(user=user)
    vector_store_id = account.vector_store_id

    try:
        thread_payload = thread_create(token, vector_store_id=vector_store_id, title=f"Audio {audio.id} Matching")
        thread_id = (thread_payload or {}).get("id") or (thread_payload or {}).get("thread_id")
        if not thread_id:
            raise ValueError("Thread creation returned no id")

        trimmed_transcript = transcript_text.strip()
        if len(trimmed_transcript) > MAX_TRANSCRIPT_CHARS:
            trimmed_transcript = trimmed_transcript[:MAX_TRANSCRIPT_CHARS]
        prompt = (
            "Given the following surgical transcript, identify which stored reference documents are most relevant. "
            "Return your answer using the documented JSON schema only. Transcript:\n" + trimmed_transcript
        )
        message_create(token, thread_id=thread_id, content=prompt, role="user")

        run_payload = run_create(token, thread_id=thread_id, assistant_id=assistant_id)
        run_id = (run_payload or {}).get("id") or (run_payload or {}).get("run_id")
        if not run_id:
            raise ValueError("Run creation returned no id")

        final_status = (run_payload or {}).get("status") or (run_payload or {}).get("state")
        if final_status not in {"completed", "failed", "requires_action"}:
            final_status = _poll_run_status(token, run_id)

        if final_status == "failed":
            raise ValueError("RAG run failed")

        assistant_message = _fetch_latest_assistant_message(token, thread_id)
        parsed = _parse_assistant_payload(assistant_message)
        result = _evaluate_matches(user, parsed)
        _persist_result(audio, result)
    except Exception as exc:  # pragma: no cover - network failure path
        logger.exception("Document matching failed for audio %s", audio.id)
        _persist_result(audio, MatchResult(status="error", documents=[], error=str(exc)))


def _poll_run_status(token: str, run_id: str) -> str:
    for _ in range(RUN_POLL_ATTEMPTS):
        time.sleep(RUN_POLL_INTERVAL_SECONDS)
        try:
            detail = run_detail(token, run_id=run_id) or {}
        except Exception:  # pragma: no cover - network failure path
            continue
        status = detail.get("status") or detail.get("state")
        if status in {"completed", "failed", "requires_action"}:
            return status
    return "unknown"


def _fetch_latest_assistant_message(token: str, thread_id: str) -> str:
    messages = thread_messages(token, thread_id=thread_id) or []
    assistant_messages = [m for m in messages if (m.get("role") or "").lower() == "assistant"]
    if not assistant_messages:
        return ""
    assistant_messages.sort(key=lambda m: m.get("created_at") or m.get("id") or 0)
    return assistant_messages[-1].get("content", "")


def _parse_assistant_payload(content: str) -> Dict[str, Any]:
    if not content:
        return {}
    stripped = content.strip()
    # Remove fenced code blocks
    if stripped.startswith("```"):
        stripped = re.sub(r"```(?:json)?", "", stripped, flags=re.IGNORECASE)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # attempt to locate first JSON object in text
    match = re.search(r"\{[\s\S]*\}", stripped)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _evaluate_matches(user: UserProfile, payload: Dict[str, Any]) -> MatchResult:
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list):
        documents = []

    normalized: List[Dict[str, Any]] = []
    for item in documents:
        if not isinstance(item, dict):
            continue
        rag_id = str(item.get("rag_document_id") or item.get("document_id") or "").strip()
        if not rag_id:
            continue
        confidence = item.get("confidence") or item.get("score")
        try:
            confidence_val = float(confidence)
        except (TypeError, ValueError):
            confidence_val = None
        name = item.get("reference_document_name") or item.get("name") or ""
        ref = ReferenceDocument.objects.filter(rag_document_id=rag_id, uploaded_by=user).first()
        normalized.append(
            {
                "rag_document_id": rag_id,
                "reference_document_name": ref.name if ref else name,
                "confidence": confidence_val,
                "reference_document_id": str(ref.id) if ref else None,
            }
        )

    normalized.sort(key=lambda d: d.get("confidence") or 0.0, reverse=True)

    if not normalized:
        return MatchResult(status="no_match", documents=[])

    top_conf = normalized[0].get("confidence") or 0.0
    similar_cutoff = max(top_conf - SIMILARITY_DELTA, 0.0)
    high_conf_docs = [d for d in normalized if (d.get("confidence") or 0.0) >= similar_cutoff]

    best_doc = normalized[0]
    best_conf = best_doc.get("confidence") or 0.0
    best_ref_id = best_doc.get("reference_document_id")

    if best_conf >= MIN_CONFIDENCE and len(high_conf_docs) == 1 and best_ref_id:
        return MatchResult(status="matched", documents=normalized, selected_reference_id=best_ref_id)

    if len(high_conf_docs) > 1:
        return MatchResult(status="needs_selection", documents=high_conf_docs)

    return MatchResult(status="low_confidence", documents=normalized)


def _persist_result(audio: AudioFile, result: MatchResult):
    updates = {
        "rag_document_match_status": result.status,
        "rag_document_matches": {
            "documents": result.documents,
            "selected_reference_document_id": result.selected_reference_id,
        },
        "rag_document_match_error": result.error,
        "rag_document_match_updated_at": timezone.now(),
    }

    if result.status == "matched" and result.selected_reference_id:
        ref = ReferenceDocument.objects.filter(id=result.selected_reference_id).first()
        if ref:
            audio.reference_document = ref
            ProcessingSession.objects.filter(audio_file=audio).update(reference_document=ref)
            updates["rag_document_matches"]["selected_reference_document_name"] = ref.name
            updates["reference_document"] = ref

    AudioFile.objects.filter(id=audio.id).update(**updates)
