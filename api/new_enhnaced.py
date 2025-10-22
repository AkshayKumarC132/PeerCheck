import json
import os
import tempfile
import uuid
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from peercheck import settings
from django.db import close_old_connections
from django.http import HttpResponse, Http404
from django.utils import timezone
from rest_framework import status
from rest_framework.generics import CreateAPIView, GenericAPIView
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import traceback

from .models import (
    ReferenceDocument,
    AudioFile,
    ProcessingSession,
    UserProfile,
    AuditLog,
    SpeakerProfile,
    RAGAssistant, RAGThread, RAGMessage, RAGRun
)
from .new_utils import (
    extract_text_from_s3,
    transcribe_audio_from_s3,
    diarization_from_audio,
    build_speaker_summary,
    find_missing,
    upload_file_to_s3,
    get_s3_key_from_url,
    s3_client,
    create_highlighted_pdf_document,
    build_three_part_communication_summary,
    get_media_duration,
)
from .authentication import token_verification
from .new_serializers import (
    UploadAndProcessSerializer,
    ProcessingResultSerializer,
    DownloadRequestSerializer,
    UserDocumentsSerializer,
    CleanupRequestSerializer,
    CleanupResponseSerializer,
    ErrorResponseSerializer,
    ProcessingSessionDetailSerializer,
    ReferenceDocumentSerializer,
    ReferenceDocumentDetailSerializer,
    ReferenceDocumentUpdateSerializer,
    AudioFileFullSerializer,
    AudioFileUpdateSerializer,
    RunDiarizationSerializer,
    SpeakerProfileMappingSerializer,
    SpeakerProfileDetailSerializer,
    SpeakerProfileCreateUpdateSerializer,
    RAGAssistantSerializer, RAGThreadSerializer, RAGMessageSerializer, RAGRunSerializer
)

# RAG helpers
from .rag_integration import ensure_user_vector_store_id, ensure_rag_token
from .rag_matching import schedule_document_match
from .ragitify_client import (
    document_ingest, document_status,
    assistant_create, assistant_list, assistant_detail,
    thread_create, thread_list, thread_detail, thread_messages,
    message_create, message_list, message_detail,
    run_create, run_list, run_detail, run_submit_tool_outputs
)

# --------------- RAG helper ---------------
def rag_feature_enabled() -> bool:
    return bool(getattr(settings, "RAGITIFY_ENABLED", False)) and bool(getattr(settings, "RAGITIFY_BASE_URL", ""))

logger = logging.getLogger(__name__)

def _bg(target, *args, **kwargs):
    t = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t

def _start_diarization_thread(
    audio_file_id,
    audio_source,
    transcript_segments,
    transcript_words,
    initiated_by_user_id=None,
    source="UploadAndProcessView",
):
    """Kick off background diarization work for the given audio file."""

    AudioFile.objects.filter(id=audio_file_id).update(diarization_status='processing')
    AuditLog.objects.create(
        action='diarization_start',
        user_id=initiated_by_user_id,
        object_id=str(audio_file_id),
        object_type='AudioFile',
        details={'source': source},
    )

    def _run_diarization():
        close_old_connections()
        try:
            segments = diarization_from_audio(audio_source, transcript_segments, transcript_words)
            payload = {"segments": segments, "speakers": build_speaker_summary(segments)}
            AudioFile.objects.filter(id=audio_file_id).update(
                diarization=payload,
                diarization_status='completed',
            )
            AuditLog.objects.create(
                action='diarization_complete',
                user_id=initiated_by_user_id,
                object_id=str(audio_file_id),
                object_type='AudioFile',
                details={'source': source, 'segments': len(segments)},
            )
        except Exception as exc:
            logger.exception("Failed to complete diarization for audio %s", audio_file_id)
            AudioFile.objects.filter(id=audio_file_id).update(
                diarization=None,
                diarization_status='failed',
            )
            AuditLog.objects.create(
                action='diarization_failed',
                user_id=initiated_by_user_id,
                object_id=str(audio_file_id),
                object_type='AudioFile',
                details={'source': source, 'error': str(exc)},
            )

    thread = threading.Thread(target=_run_diarization, daemon=True)
    thread.start()
    return thread


def _build_document_option(reference: ReferenceDocument, confidence: Optional[float], rag_document_id: Optional[str]) -> Dict[str, Any]:
    return {
        "reference_document_id": str(reference.id),
        "name": reference.name,
        "rag_document_id": rag_document_id or reference.rag_document_id,
        "confidence": confidence,
        "file_path": reference.file_path,
        "document_type": reference.document_type,
    }


def _serialize_rag_candidates(user: UserProfile, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    for item in candidates:
        rag_doc_id = item.get("rag_document_id")
        ref_id = item.get("reference_document_id")
        confidence = item.get("confidence")
        reference = None
        if ref_id:
            reference = ReferenceDocument.objects.filter(id=ref_id, uploaded_by=user).first()
        elif rag_doc_id:
            reference = ReferenceDocument.objects.filter(rag_document_id=rag_doc_id, uploaded_by=user).first()
        if reference:
            options.append(_build_document_option(reference, confidence, rag_doc_id))
        else:
            options.append({
                "reference_document_id": None,
                "rag_document_id": rag_doc_id,
                "reference_document_name": item.get("reference_document_name"),
                "confidence": confidence,
            })
    return options


def _all_user_document_options(user: UserProfile) -> List[Dict[str, Any]]:
    docs = ReferenceDocument.objects.filter(uploaded_by=user)
    return [_build_document_option(doc, None, doc.rag_document_id) for doc in docs]

# --------------------------- CORE: Upload + Process ---------------------------

class UploadAndProcessView(CreateAPIView):
    """
    Upload text and audio files to S3, process them, and return analysis results
    """
    serializer_class = UploadAndProcessSerializer
    
    @swagger_auto_schema(
        operation_description="Upload and process text document with audio file",
        responses={200: ProcessingResultSerializer, 400: ErrorResponseSerializer, 401: ErrorResponseSerializer, 500: ErrorResponseSerializer}
    )
    def create(self, request, token, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error'], 'timestamp': timezone.now()}, status=status.HTTP_401_UNAUTHORIZED)
        
        user = auth_result['user']
        serializer = self.serializer_class(data=request.data)
        if not serializer.is_valid():
            return Response({'error': 'Invalid input data', 'details': serializer.errors, 'timestamp': timezone.now()}, status=status.HTTP_400_BAD_REQUEST)
        
        start_time = time.time()
        reference_doc = None
        audio_obj = None
        text_content: Optional[str] = None

        try:
            validated = serializer.validated_data
            user_profile = user
            audio_file = validated['audio_file']
            text_file = validated.get('text_file')
            existing_document_id = validated.get('existing_document_id')

            # Create or fetch ReferenceDocument when provided
            if text_file:
                document_type = validated.get('document_type', 'sop')
                document_name = validated.get('document_name', '')
                text_s3_key = f'documents/{user.id}/{uuid.uuid4()}_{text_file.name}'
                text_s3_url = upload_file_to_s3(text_file, text_s3_key)

                reference_doc = ReferenceDocument.objects.create(
                    name=document_name or text_file.name,
                    document_type=document_type,
                    file_path=text_s3_url,
                    original_filename=text_file.name,
                    file_size=text_file.size,
                    content_type=getattr(text_file, "content_type", None) or 'application/octet-stream',
                    uploaded_by=user_profile,
                    upload_status='processing'
                )

                text_content = extract_text_from_s3(text_s3_url)
                reference_doc.extracted_text = text_content
                reference_doc.upload_status = 'processed'
                reference_doc.save(update_fields=['extracted_text', 'upload_status'])
                AuditLog.objects.create(
                    action='document_upload',
                    user=user_profile,
                    object_id=str(reference_doc.id),
                    object_type='ReferenceDocument',
                    details={
                        'document_type': reference_doc.document_type,
                        'original_filename': reference_doc.original_filename,
                        'file_size': reference_doc.file_size,
                        'source': 'UploadAndProcessView',
                    },
                )

            elif existing_document_id:
                try:
                    reference_doc = ReferenceDocument.objects.get(id=existing_document_id, uploaded_by=user_profile)
                except ReferenceDocument.DoesNotExist:
                    return Response({
                        'error': 'Reference document not found',
                        'details': {'existing_document_id': [f"No ReferenceDocument with ID '{existing_document_id}'."]},
                        'timestamp': timezone.now()
                    }, status=status.HTTP_404_NOT_FOUND)

                text_content = reference_doc.extracted_text
                if (not text_content) and reference_doc.file_path:
                    text_content = extract_text_from_s3(reference_doc.file_path)
                    reference_doc.extracted_text = text_content
                    reference_doc.save(update_fields=['extracted_text'])

            # --- RAG ingestion (document) ---
            if reference_doc and rag_feature_enabled():
                token_val, vs_id, err_vs = ensure_user_vector_store_id(user)
                if token_val and vs_id:
                    try:
                        # store the user’s VS id into the document if not already set
                        if not reference_doc.rag_vector_store_id:
                            reference_doc.rag_enabled = True
                            reference_doc.rag_vector_store_id = vs_id
                            reference_doc.save(update_fields=["rag_enabled", "rag_vector_store_id"])

                        # ingest document by S3 URL
                        if reference_doc.rag_vector_store_id and not reference_doc.rag_document_id:
                            ingest = document_ingest(token_val, vector_store_id=reference_doc.rag_vector_store_id, s3_file_url=reference_doc.file_path)
                            doc_id = (ingest or {}).get("id") or (ingest or {}).get("document_id")
                            status_str = (ingest or {}).get("status") or "queued"
                            reference_doc.rag_document_id = str(doc_id) if doc_id else None
                            reference_doc.rag_status = status_str
                            reference_doc.rag_uploaded_at = timezone.now()
                            reference_doc.rag_last_error = None
                            reference_doc.rag_metadata = ingest or {}
                            reference_doc.save(update_fields=["rag_document_id","rag_status","rag_uploaded_at","rag_last_error","rag_metadata"])

                            # non-blocking status check (one shot)
                            try:
                                st = document_status(token_val, document_id=str(doc_id)) or {}
                                s = st.get("status") or st.get("state")
                                if s:
                                    reference_doc.rag_status = s
                                if (s or "").lower() in ("completed", "ready", "processed"):
                                    reference_doc.rag_ingested_at = timezone.now()
                                if st.get("error"):
                                    reference_doc.rag_last_error = st.get("error")
                                reference_doc.rag_metadata = {**(reference_doc.rag_metadata or {}), "status_check": st}
                                reference_doc.save(update_fields=["rag_status","rag_ingested_at","rag_last_error","rag_metadata"])
                            except Exception:
                                pass
                    except Exception as e:
                        reference_doc.rag_enabled = True
                        reference_doc.rag_last_error = str(e)
                        reference_doc.save(update_fields=["rag_enabled","rag_last_error"])

            # --- New: Get media duration before upload ---
            audio_file.seek(0)
            audio_duration = get_media_duration(audio_file)
            audio_file.seek(0)  # reset for upload

            # Upload audio file
            audio_s3_key = f'audio/{user.id}/{uuid.uuid4()}_{audio_file.name}'
            audio_s3_url = upload_file_to_s3(audio_file, audio_s3_key)
            audio_obj = AudioFile.objects.create(
                file_path=audio_s3_url,
                original_filename=audio_file.name,
                user=user_profile,
                reference_document=reference_doc,
                status='processing',
                duration=audio_duration,
            )
            AuditLog.objects.create(
                action='audio_upload',
                user=user_profile,
                object_id=str(audio_obj.id),
                object_type='AudioFile',
                details={
                    'original_filename': audio_obj.original_filename,
                    'reference_document_id': str(reference_doc.id) if reference_doc else None,
                    'source': 'UploadAndProcessView',
                },
            )

            # Transcribe audio
            try:
                transcript_result = transcribe_audio_from_s3(audio_s3_url)
                transcript = transcript_result["text"]
                transcript_segments = transcript_result.get("segments", [])
                transcript_words = transcript_result.get("words", [])
                audio_obj.transcription = transcript_result
                audio_obj.save()
            except ValueError as ve:
                logging.error(f"Transcription error: {ve}")
                return Response({
                    'error': 'Failed to transcribe audio. Please check the audio file.',
                    'details': str(ve),
                    'timestamp': timezone.now().isoformat()
                }, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                logging.exception("Unexpected error during transcription")
                return Response({
                    'error': 'An unexpected error occurred during transcription.',
                    'details': str(e),
                    'timestamp': timezone.now().isoformat()
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Speaker diarization
            audio_obj.diarization = None
            audio_obj.save(update_fields=['diarization'])
            _start_diarization_thread(
                audio_obj.id,
                audio_s3_url,
                transcript_segments,
                transcript_words,
                initiated_by_user_id=user_profile.id,
                source='UploadAndProcessView',
            )
            audio_obj.diarization_status = 'processing'
            
            matched_html = ''
            missing_html = ''
            entire_html = ''
            matched_words = 0
            total_words = 0
            coverage = None

            if text_content:
                matched_html, missing_html, matched_words, total_words, entire_html = find_missing(text_content, transcript)
                coverage = (matched_words / total_words * 100) if total_words > 0 else 0
                audio_obj.coverage = coverage
            else:
                audio_obj.coverage = None

            audio_obj.status = 'processed'
            audio_obj.save()
            AuditLog.objects.create(
                action='audio_process',
                user=user_profile,
                object_id=str(audio_obj.id),
                object_type='AudioFile',
                details={
                    'status': audio_obj.status,
                    'coverage': coverage,
                    'reference_document_id': str(reference_doc.id) if reference_doc else None,
                },
            )

            # Kick off asynchronous document matching with RAGitify
            schedule_document_match(audio_obj)

            # Create processing session
            expires_at = timezone.now() + timedelta(hours=24)
            session = ProcessingSession.objects.create(
                reference_document=reference_doc,
                audio_file=audio_obj,
                matched_words=matched_words,
                total_words=total_words,
                coverage=coverage if coverage is not None else 0,
                expires_at=expires_at
            )
            processing_time = time.time() - start_time
            audio_obj.processing_session = session.id
            audio_obj.save()

            return Response({
                'session_id': str(session.id),
                'matched_words': matched_words,
                'total_words': total_words,
                'coverage': round(coverage, 2) if coverage is not None else None,
                'reference_document_id': str(reference_doc.id) if reference_doc else None,
                'audio_file_id': str(audio_obj.id),
                'matched_content': matched_html if text_content else None,
                'missing_content': missing_html if text_content else None,
                'entire_document': entire_html if text_content else None,
                'processing_time': round(processing_time, 2),
                'diarization': None,
                'diarization_error': None,
                'diarization_status': audio_obj.diarization_status,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            if reference_doc:
                reference_doc.upload_status = 'failed'
                reference_doc.save()
            if audio_obj:
                audio_obj.status = 'failed'
                audio_obj.diarization_status = 'failed'
                audio_obj.save(update_fields=['status', 'diarization_status'])
                AuditLog.objects.create(
                    action='diarization_failed',
                    user=user_profile,
                    object_id=str(audio_obj.id),
                    object_type='AudioFile',
                    details={'source': 'UploadAndProcessView', 'error': str(e)},
                )
            logger.exception("UploadAndProcessView failed")
            return Response({'error': f'Processing failed: {str(e)}','timestamp': timezone.now().isoformat()}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ---------------- Other original endpoints (kept intact) ----------------

class RunDiarizationView(CreateAPIView):
    """API to rerun speaker diarization for a processed audio file."""

    serializer_class = RunDiarizationSerializer

    def create(self, request, token, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({
                'error': auth_result['error'],
                'timestamp': timezone.now(),
            }, status=status.HTTP_401_UNAUTHORIZED)

        user = auth_result['user']

        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        audio_id = serializer.validated_data['audio_id']

        try:
            audio_file = AudioFile.objects.get(id=audio_id)
        except AudioFile.DoesNotExist:
            return Response({
                'error': 'Audio file not found',
                'audio_id': str(audio_id),
            }, status=status.HTTP_404_NOT_FOUND)

        transcript = audio_file.transcription or {}
        transcript_segments = transcript.get('segments', [])
        transcript_words = transcript.get('words', [])

        audio_file.diarization = None
        audio_file.save(update_fields=['diarization'])

        _start_diarization_thread(
            audio_file.id,
            audio_file.file_path,
            transcript_segments,
            transcript_words,
            initiated_by_user_id=user.id,
            source='RunDiarizationView',
        )
        audio_file.diarization_status = 'processing'
        audio_file.save(update_fields=['diarization_status'])

        return Response({
            'audio_file_id': str(audio_file.id),
            'status': 'processing',
            'diarization_status': audio_file.diarization_status,
        }, status=status.HTTP_202_ACCEPTED)


class SpeakerProfileMappingView(CreateAPIView):
    """API to assign or update speaker profiles using diarization results."""

    serializer_class = SpeakerProfileMappingSerializer

    def create(self, request, token, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({
                'error': auth_result['error'],
                'timestamp': timezone.now(),
            }, status=status.HTTP_401_UNAUTHORIZED)

        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        audio_id = serializer.validated_data['audio_id']
        speaker_label = serializer.validated_data['speaker_label']
        name = serializer.validated_data['name'].strip()
        profile_id = serializer.validated_data.get('profile_id')

        try:
            audio_file = AudioFile.objects.get(id=audio_id)
        except AudioFile.DoesNotExist:
            return Response({
                'error': 'Audio file not found',
                'audio_id': str(audio_id),
            }, status=status.HTTP_404_NOT_FOUND)

        diarization_data = audio_file.diarization or {}
        if isinstance(diarization_data, dict):
            segments = diarization_data.get('segments', [])
        else:
            segments = diarization_data

        if not segments:
            return Response({
                'error': 'No diarization data available for this audio file',
                'audio_file_id': str(audio_file.id),
            }, status=status.HTTP_400_BAD_REQUEST)

        matched_segments = [
            seg for seg in segments
            if (seg.get('speaker_label') or seg.get('speaker')) == speaker_label
        ]

        if not matched_segments:
            return Response({
                'error': f'Speaker label {speaker_label} not found in diarization results',
                'audio_file_id': str(audio_file.id),
            }, status=status.HTTP_404_NOT_FOUND)

        vectors = [
            np.array(seg.get('speaker_vector'), dtype=float)
            for seg in matched_segments if seg.get('speaker_vector')
        ]
        vectors = [
            vec for vec in vectors
            if vec.size and not np.isnan(vec).any() and not np.isinf(vec).any()
        ]

        if not vectors:
            return Response({
                'error': 'Speaker embeddings are not available for the selected label',
                'audio_file_id': str(audio_file.id),
            }, status=status.HTTP_400_BAD_REQUEST)

        mean_vector = np.mean(vectors, axis=0)
        if mean_vector is None or np.isnan(mean_vector).any():
            return Response({
                'error': 'Failed to compute a valid speaker embedding for the selected label',
                'audio_file_id': str(audio_file.id),
            }, status=status.HTTP_400_BAD_REQUEST)

        mean_vector_list = mean_vector.tolist()

        if not profile_id:
            profile_id = next(
                (seg.get('speaker_profile_id') for seg in matched_segments if seg.get('speaker_profile_id')),
                None,
            )

        created_profile = False

        try:
            if profile_id:
                profile = SpeakerProfile.objects.get(id=profile_id)
                profile.embedding = mean_vector_list
                profile.name = name or profile.name
                profile.save()
            else:
                profile = SpeakerProfile.objects.filter(name__iexact=name).first()
                if profile:
                    profile.embedding = mean_vector_list
                    profile.name = name
                    profile.save()
                else:
                    profile = SpeakerProfile.objects.create(
                        name=name or speaker_label,
                        embedding=mean_vector_list,
                    )
                    created_profile = True
        except SpeakerProfile.DoesNotExist:
            return Response({
                'error': 'Specified speaker profile does not exist',
                'profile_id': profile_id,
            }, status=status.HTTP_404_NOT_FOUND)

        updated_segments = []
        for seg in segments:
            seg_label = seg.get('speaker_label') or seg.get('speaker')
            if seg_label == speaker_label:
                seg['speaker'] = profile.name or speaker_label
                seg['speaker_name'] = profile.name
                seg['speaker_profile_id'] = profile.id
            updated_segments.append(seg)

        diarization_payload = {
            'segments': updated_segments,
            'speakers': build_speaker_summary(updated_segments),
        }
        audio_file.diarization = diarization_payload
        audio_file.save(update_fields=['diarization'])
        AuditLog.objects.create(
            action='speaker_profile_create' if created_profile else 'speaker_profile_update',
            user=user,
            object_id=str(profile.id),
            object_type='SpeakerProfile',
            details={
                'audio_file_id': str(audio_file.id),
                'speaker_label': speaker_label,
                'profile_name': profile.name,
            },
        )

        return Response({
            'audio_file_id': str(audio_file.id),
            'speaker_profile': {
                'id': profile.id,
                'name': profile.name,
            },
            'diarization': diarization_payload,
        }, status=status.HTTP_200_OK)

class SpeakerProfileListCreateView(GenericAPIView):
    serializer_class = SpeakerProfileDetailSerializer

    def get(self, request, token, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        profiles = SpeakerProfile.objects.all().order_by('name')
        serializer = self.serializer_class(profiles, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, token, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        user = auth_result['user']
        serializer = SpeakerProfileCreateUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        profile = serializer.save()

        AuditLog.objects.create(
            action='speaker_profile_create',
            user=user,
            object_id=str(profile.id),
            object_type='SpeakerProfile',
            details={
                'name': profile.name,
                'source': 'SpeakerProfileListCreateView',
            },
        )

        response_serializer = self.serializer_class(profile)
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)

class SpeakerProfileDetailView(GenericAPIView):
    serializer_class = SpeakerProfileDetailSerializer

    def _get_profile(self, profile_id):
        try:
            return SpeakerProfile.objects.get(id=profile_id)
        except SpeakerProfile.DoesNotExist:
            raise Http404('Speaker profile not found')

    def get(self, request, token, profile_id, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        profile = self._get_profile(profile_id)
        serializer = self.serializer_class(profile)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, token, profile_id, *args, **kwargs):
        return self._update(request, token, profile_id, partial=False)

    def patch(self, request, token, profile_id, *args, **kwargs):
        return self._update(request, token, profile_id, partial=True)

    def _update(self, request, token, profile_id, partial):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        user = auth_result['user']
        profile = self._get_profile(profile_id)

        serializer = SpeakerProfileCreateUpdateSerializer(
            instance=profile,
            data=request.data,
            partial=partial,
        )
        serializer.is_valid(raise_exception=True)

        original = {
            'name': profile.name,
            'embedding': profile.embedding,
        }

        profile = serializer.save()
        profile.refresh_from_db()

        updated = {
            'name': profile.name,
            'embedding': profile.embedding,
        }

        changes = {}
        for field, original_value in original.items():
            if updated.get(field) != original_value:
                changes[field] = {'from': original_value, 'to': updated.get(field)}

        AuditLog.objects.create(
            action='speaker_profile_update',
            user=user,
            object_id=str(profile.id),
            object_type='SpeakerProfile',
            details={
                'changes': changes,
                'source': 'SpeakerProfileDetailView',
            },
        )

        response_serializer = self.serializer_class(profile)
        return Response(response_serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, token, profile_id, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        user = auth_result['user']
        profile = self._get_profile(profile_id)

        profile_id_str = str(profile.id)
        profile_name = profile.name
        profile.delete()

        AuditLog.objects.create(
            action='speaker_profile_delete',
            user=user,
            object_id=profile_id_str,
            object_type='SpeakerProfile',
            details={
                'name': profile_name,
                'source': 'SpeakerProfileDetailView',
            },
        )

        return Response(status=status.HTTP_204_NO_CONTENT)

class DownloadProcessedDocumentView(GenericAPIView):
    """
    Download the processed DOCX with highlighted text
    """
    serializer_class = DownloadRequestSerializer
    #authentication_classes = [KnoxTokenAuthentication]
    
    @swagger_auto_schema(
        operation_description="Download processed document with highlighted text",
        responses={
            200: openapi.Response(
                description="Processed DOCX file",
                content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ),
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
            500: ErrorResponseSerializer
        }
    )
    def get(self, request, token, session_id, *args, **kwargs):
        # Verify token
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({
                'error': user_data['error'],
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_401_UNAUTHORIZED)
        
        user = user_data['user']
        print(f"User: {user.username}, Session ID: {session_id}")
        
        try:
            # Get processing session
            session = ProcessingSession.objects.get(
                id=session_id
            )
            
            reference_doc = session.reference_document
            audio_file = session.audio_file

            if rag_feature_enabled():
                match_status = audio_file.rag_document_match_status or ""
                match_payload = audio_file.rag_document_matches or {}

                if match_status in ("", None, "pending"):
                    return Response({
                        'status': 'matching',
                        'message': 'Automatic document matching is still running. Please retry shortly.',
                        'timestamp': timezone.now().isoformat(),
                    }, status=status.HTTP_202_ACCEPTED)

                if match_status == "matched":
                    selected_id = match_payload.get("selected_reference_document_id")
                    if selected_id:
                        resolved = ReferenceDocument.objects.filter(id=selected_id, uploaded_by=user).first()
                        if resolved:
                            reference_doc = resolved

                elif match_status == "needs_selection":
                    options = _serialize_rag_candidates(user, match_payload.get("documents", []))
                    return Response({
                        'status': 'needs_selection',
                        'message': 'Multiple documents closely match this transcript. Please select the correct reference document.',
                        'options': options,
                        'timestamp': timezone.now().isoformat(),
                    }, status=status.HTTP_409_CONFLICT)

                elif match_status in ("no_match", "low_confidence"):
                    options = _all_user_document_options(user)
                    return Response({
                        'status': 'manual_selection_required',
                        'message': 'We could not confidently identify the reference document. Please choose from your available documents.',
                        'options': options,
                        'timestamp': timezone.now().isoformat(),
                    }, status=status.HTTP_409_CONFLICT)

                elif match_status == "error":
                    options = _all_user_document_options(user)
                    return Response({
                        'status': 'manual_selection_required',
                        'message': f"Automatic document matching failed: {audio_file.rag_document_match_error or 'Unknown error'}. Please select the correct document manually.",
                        'options': options,
                        'timestamp': timezone.now().isoformat(),
                    }, status=status.HTTP_409_CONFLICT)

            if reference_doc is None:
                options = _all_user_document_options(user)
                return Response({
                    'status': 'manual_selection_required',
                    'message': 'No reference document is linked to this audio. Please select one to continue.',
                    'options': options,
                    'timestamp': timezone.now().isoformat(),
                }, status=status.HTTP_409_CONFLICT)

            if session.reference_document_id != reference_doc.id or not session.processed_docx_path:
                # ensure session reference matches and clear stale processed files when switching documents
                if session.reference_document_id != reference_doc.id:
                    session.reference_document = reference_doc
                    session.processed_docx_path = None
                    session.save(update_fields=['reference_document', 'processed_docx_path'])
                else:
                    session.reference_document = reference_doc
                    session.save(update_fields=['reference_document'])

            if audio_file.reference_document_id != reference_doc.id:
                audio_file.reference_document = reference_doc
                audio_file.save(update_fields=['reference_document'])

            # Check if processed document already exists in S3
            previous_processed_url = session.processed_docx_path
            processed_s3_url = (
                previous_processed_url
                if previous_processed_url and previous_processed_url.lower().endswith('.pdf')
                else None
            )

            if processed_s3_url:
                print(f"Using existing processed document: {processed_s3_url}")
            else:
                print("Creating new processed document...")
                try:
                    # Create highlighted PDF and upload to S3
                    transcript = audio_file.transcription.get('text', '') if audio_file.transcription else ''

                    if not transcript:
                        return Response({
                            'error': 'No transcript available for processing',
                            'timestamp': timezone.now().isoformat()
                        }, status=status.HTTP_400_BAD_REQUEST)
                    
                    # Create highlighted document based on file extension
                    file_ext = reference_doc.file_path.rsplit('.', 1)[-1].lower()
                    use_transcript = (
                        request.query_params.get("validate_abbreviations", "true")
                        .lower()
                        == "true"
                    )
                    logging.info(
                        "validate_abbreviations=%s in DownloadProcessedDocumentView",
                        use_transcript,
                    )
                    if file_ext not in ('pdf', 'docx'):
                        return Response({
                            'error': f'Unsupported file type for highlighting: {file_ext}',
                            'timestamp': timezone.now().isoformat()
                        }, status=status.HTTP_400_BAD_REQUEST)

                    processed_s3_url = create_highlighted_pdf_document(
                        reference_doc.file_path,
                        transcript,
                        require_transcript_match=use_transcript,
                    )
                    print(f"Created processed document: {processed_s3_url}")
                    # Save S3 URL to session
                    session.processed_docx_path = processed_s3_url
                    session.save()

                    if previous_processed_url and previous_processed_url != processed_s3_url:
                        try:
                            if previous_processed_url.startswith('s3://') or ('amazonaws.com/' in previous_processed_url):
                                s3_key = get_s3_key_from_url(previous_processed_url)
                                s3_client.delete_object(
                                    Bucket=settings.AWS_STORAGE_BUCKET_NAME,
                                    Key=s3_key,
                                )
                        except Exception:
                            logger.info("Unable to delete previous processed document %s", previous_processed_url)
                    
                except Exception as doc_error:
                    print(f"Error creating highlighted document: {str(doc_error)}")
                    print(traceback.format_exc())
                    return Response({
                        'error': f'Failed to create processed document: {str(doc_error)}',
                        'timestamp': timezone.now().isoformat()
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Save the S3 link to AudioFile.report_path
            audio_file.report_path = processed_s3_url
            audio_file.save()

            AuditLog.objects.create(
                action='document_download',
                user=user,
                object_id=str(reference_doc.id) if reference_doc else str(session.id),
                object_type='ReferenceDocument' if reference_doc else 'ProcessingSession',
                details={
                    'session_id': str(session.id),
                    'audio_file_id': str(audio_file.id),
                    'with_diarization': False,
                    'source': 'DownloadProcessedDocumentView',
                },
            )

            # Direct file download
            return Response({
                'processed_docx_url': processed_s3_url,
                'message': 'Processed document is available at the above URL.',
                'session_id': str(session.id),
                'filename': f"{reference_doc.original_filename.rsplit('.', 1)[0]}_processed.pdf"
            }, status=status.HTTP_200_OK)
        except ProcessingSession.DoesNotExist:
            print(f"Session not found or expired: {session_id}")
            return Response({
                'error': 'Processing session not found or expired',
                'session_id': str(session_id),
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_404_NOT_FOUND)
            
        except Exception as e:
            print(f"Unexpected error in DownloadProcessedDocumentView: {str(e)}")
            print(traceback.format_exc())
            
            return Response({
                'error': f'Download failed: {str(e)}',
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DownloadProcessedDocumentWithDiarizationView(GenericAPIView):
    """Download highlighted document enriched with speaker diarization details."""

    serializer_class = DownloadRequestSerializer

    @swagger_auto_schema(
        operation_description="Download processed document with speaker diarization summary",
        responses={
            200: openapi.Response(
                description="Processed DOCX file with 3PC details",
                content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ),
            202: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
            409: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        }
    )
    def get(self, request, token, session_id, *args, **kwargs):
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({
                'error': user_data['error'],
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_401_UNAUTHORIZED)

        user = user_data['user']
        print(f"User: {user.username}, Session ID: {session_id}")

        try:
            session = ProcessingSession.objects.get(id=session_id)

            reference_doc = session.reference_document
            audio_file = session.audio_file

            if rag_feature_enabled():
                match_status = audio_file.rag_document_match_status or ""
                match_payload = audio_file.rag_document_matches or {}

                if match_status in ("", None, "pending"):
                    return Response({
                        'status': 'matching',
                        'message': 'Automatic document matching is still running. Please retry shortly.',
                        'timestamp': timezone.now().isoformat(),
                    }, status=status.HTTP_202_ACCEPTED)

                if match_status == "matched":
                    selected_id = match_payload.get("selected_reference_document_id")
                    if selected_id:
                        resolved = ReferenceDocument.objects.filter(id=selected_id, uploaded_by=user).first()
                        if resolved:
                            reference_doc = resolved

                elif match_status == "needs_selection":
                    options = _serialize_rag_candidates(user, match_payload.get("documents", []))
                    return Response({
                        'status': 'needs_selection',
                        'message': 'Multiple documents closely match this transcript. Please select the correct reference document.',
                        'options': options,
                        'timestamp': timezone.now().isoformat(),
                    }, status=status.HTTP_409_CONFLICT)

                elif match_status in ("no_match", "low_confidence"):
                    options = _all_user_document_options(user)
                    return Response({
                        'status': 'manual_selection_required',
                        'message': 'We could not confidently identify the reference document. Please choose from your available documents.',
                        'options': options,
                        'timestamp': timezone.now().isoformat(),
                    }, status=status.HTTP_409_CONFLICT)

                elif match_status == "error":
                    options = _all_user_document_options(user)
                    return Response({
                        'status': 'manual_selection_required',
                        'message': f"Automatic document matching failed: {audio_file.rag_document_match_error or 'Unknown error'}. Please select the correct document manually.",
                        'options': options,
                        'timestamp': timezone.now().isoformat(),
                    }, status=status.HTTP_409_CONFLICT)

            if reference_doc is None:
                options = _all_user_document_options(user)
                return Response({
                    'status': 'manual_selection_required',
                    'message': 'No reference document is linked to this audio. Please select one to continue.',
                    'options': options,
                    'timestamp': timezone.now().isoformat(),
                }, status=status.HTTP_409_CONFLICT)

            if session.reference_document_id != reference_doc.id:
                session.reference_document = reference_doc
                session.processed_docx_with_diarization_path = None
                session.save(update_fields=['reference_document', 'processed_docx_with_diarization_path'])

            if audio_file.reference_document_id != reference_doc.id:
                audio_file.reference_document = reference_doc
                audio_file.save(update_fields=['reference_document'])

            transcript = audio_file.transcription.get('text', '') if audio_file.transcription else ''
            if not transcript:
                return Response({
                    'error': 'No transcript available for processing',
                    'timestamp': timezone.now().isoformat()
                }, status=status.HTTP_400_BAD_REQUEST)

            diarization_payload = audio_file.diarization
            if diarization_payload is None:
                return Response({
                    'status': 'diarization_pending',
                    'message': 'Speaker diarization is still processing. Please try again shortly.',
                    'timestamp': timezone.now().isoformat(),
                }, status=status.HTTP_202_ACCEPTED)

            if isinstance(diarization_payload, dict):
                diarization_segments = diarization_payload.get('segments', [])
            else:
                diarization_segments = diarization_payload

            reference_text = reference_doc.extracted_text or ''
            if not reference_text and reference_doc.file_path:
                try:
                    reference_text = extract_text_from_s3(reference_doc.file_path)
                except Exception as extraction_error:
                    logger.warning("Failed to refresh reference text for 3PC summary: %s", extraction_error)
                    reference_text = ''
                else:
                    reference_doc.extracted_text = reference_text
                    reference_doc.save(update_fields=['extracted_text'])

            three_pc_entries = build_three_part_communication_summary(reference_text, diarization_segments)

            try:
                previous_url = session.processed_docx_with_diarization_path
                reusable_url = (
                    previous_url
                    if previous_url and previous_url.lower().endswith('.pdf')
                    else None
                )

                if reusable_url:
                    print(f"Using existing diarization document: {reusable_url}")
                    processed_s3_url = reusable_url
                else:
                    processed_s3_url = create_highlighted_pdf_document(
                        reference_doc.file_path,
                        transcript,
                        three_pc_entries=three_pc_entries,
                    )

                if previous_url and previous_url != processed_s3_url:
                    try:
                        if previous_url.startswith('s3://') or ('amazonaws.com/' in previous_url):
                            s3_key = get_s3_key_from_url(previous_url)
                            s3_client.delete_object(
                                Bucket=settings.AWS_STORAGE_BUCKET_NAME,
                                Key=s3_key,
                            )
                    except Exception:
                        logger.info("Unable to delete previous diarization document %s", previous_url)

                session.processed_docx_with_diarization_path = processed_s3_url
                session.save(update_fields=['processed_docx_with_diarization_path'])

            except Exception as doc_error:
                print(f"Error creating diarization document: {str(doc_error)}")
                print(traceback.format_exc())
                return Response({
                    'error': f'Failed to create processed document with diarization: {str(doc_error)}',
                    'timestamp': timezone.now().isoformat()
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            response_payload = {
                'processed_docx_url': processed_s3_url,
                'message': 'Processed document with speaker details is available at the above URL.',
                'session_id': str(session.id),
                'filename': f"{reference_doc.original_filename.rsplit('.', 1)[0]}_processed_diarization.pdf",
                'three_pc_entries': three_pc_entries,
            }

            AuditLog.objects.create(
                action='document_download',
                user=user,
                object_id=str(reference_doc.id) if reference_doc else str(session.id),
                object_type='ReferenceDocument' if reference_doc else 'ProcessingSession',
                details={
                    'session_id': str(session.id),
                    'audio_file_id': str(audio_file.id),
                    'with_diarization': True,
                    'source': 'DownloadProcessedDocumentWithDiarizationView',
                },
            )

            return Response(response_payload, status=status.HTTP_200_OK)

        except ProcessingSession.DoesNotExist:
            print(f"Session not found or expired: {session_id}")
            return Response({
                'error': 'Processing session not found or expired',
                'session_id': str(session_id),
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            print(f"Unexpected error in DownloadProcessedDocumentWithDiarizationView: {str(e)}")
            print(traceback.format_exc())
            return Response({
                'error': f'Download with diarization failed: {str(e)}',
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GetUserDocumentsView(GenericAPIView):
    """
    Get user's uploaded documents and audio files
    """
    serializer_class = UserDocumentsSerializer
    #authentication_classes = [KnoxTokenAuthentication]
    
    @swagger_auto_schema(
        operation_description="Get list of user's uploaded documents and audio files",
        responses={
            200: UserDocumentsSerializer,
            401: ErrorResponseSerializer
        }
    )
    def get(self, request, token, *args, **kwargs):
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)
        print(f"User data: {user_data}"
              f"Request data: {request.data}")
        user = user_data['user']
        
        try:
            user_profile = UserProfile.objects.get(**{UserProfile.USERNAME_FIELD: getattr(user, UserProfile.USERNAME_FIELD)})
            
            documents = ReferenceDocument.objects.filter(uploaded_by=user_profile)
            audio_files = AudioFile.objects.filter(user=user_profile)
            
            response_data = {
                'documents': documents,
                'audio_files': audio_files,
                'total_documents': documents.count(),
                'total_audio_files': audio_files.count()
            }
            
            serializer = self.serializer_class(response_data)
            return Response(serializer.data, status=status.HTTP_200_OK)
            
        except UserProfile.DoesNotExist:
            response_data = {
                'documents': [],
                'audio_files': [],
                'total_documents': 0,
                'total_audio_files': 0
            }
            serializer = self.serializer_class(response_data)
            return Response(serializer.data, status=status.HTTP_200_OK)

class CleanupExpiredSessionsView(CreateAPIView):
    """
    Clean up expired processing sessions and their S3 files
    """
    serializer_class = CleanupRequestSerializer
    #authentication_classes = [KnoxTokenAuthentication]
    
    @swagger_auto_schema(
        operation_description="Clean up expired processing sessions (Admin only)",
        responses={
            200: CleanupResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer
        }
    )
    def create(self, request, token, *args, **kwargs):
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)
        print(f"User data: {user_data}"
              f"Request data: {request.data}")
        user = user_data['user']
        
        if not user.is_staff:
            error_data = {
                'error': 'Permission denied - Admin access required',
                'timestamp': timezone.now()
            }
            return Response(error_data, status=status.HTTP_403_FORBIDDEN)
        
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        force_cleanup = serializer.validated_data.get('force_cleanup', False)
        
        # Get expired sessions
        query = ProcessingSession.objects.filter(expires_at__lt=timezone.now())
        if force_cleanup:
            # Include all sessions if force cleanup
            query = ProcessingSession.objects.all()
        
        expired_sessions = query
        deleted_count = 0
        cleaned_files = 0
        
        for session in expired_sessions:
            # Delete processed DOCX files from S3
            for attr in ('processed_docx_path', 'processed_docx_with_diarization_path'):
                doc_url = getattr(session, attr, None)
                if not doc_url:
                    continue
                s3_key = None
                try:
                    s3_key = get_s3_key_from_url(doc_url)
                    s3_client.delete_object(
                        Bucket=settings.AWS_STORAGE_BUCKET_NAME,
                        Key=s3_key
                    )
                    cleaned_files += 1
                except Exception as e:
                    print(f"Failed to delete S3 object {s3_key or doc_url}: {str(e)}")
            deleted_count += 1
        
        # expired_sessions.delete()
        
        response_data = {
            'message': f'Successfully cleaned up {deleted_count} expired sessions',
            'deleted_count': deleted_count,
            'cleaned_files': cleaned_files
        }
        
        response_serializer = CleanupResponseSerializer(data=response_data)
        response_serializer.is_valid()
        
        return Response(response_serializer.data, status=status.HTTP_200_OK)

class GetProcessingSessionView(GenericAPIView):
    """
    Get details of a specific processing session
    """
    serializer_class = ProcessingSessionDetailSerializer
    #authentication_classes = [KnoxTokenAuthentication]
    
    @swagger_auto_schema(
        operation_description="Get processing session details",
        responses={
            200: ProcessingSessionDetailSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer
        }
    )
    def get(self, request, token, session_id, *args, **kwargs):
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)
        print(f"User data: {user_data}"
              f"Request data: {request.data}")
        user = user_data['user']
        
        try:
            session = ProcessingSession.objects.get(id=session_id)
            
            # Verify user has access
            if session.audio_file.user != user:
                error_data = {
                    'error': 'Access denied',
                    'timestamp': timezone.now()
                }
                error_serializer = ErrorResponseSerializer(data=error_data)
                error_serializer.is_valid()
                return Response(error_serializer.data, status=status.HTTP_403_FORBIDDEN)
            
            serializer = self.serializer_class(session)
            return Response(serializer.data, status=status.HTTP_200_OK)
            
        except ProcessingSession.DoesNotExist:
            error_data = {
                'error': 'Processing session not found',
                'timestamp': timezone.now()
            }
            return Response(error_data, status=status.HTTP_404_NOT_FOUND)

class UploadReferenceDocumentView(CreateAPIView):
    serializer_class = ReferenceDocumentSerializer

    def post(self, request, token, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=401)
        user = auth_result['user']

        # ---- input & upload to S3 ----
        try:
            f = request.FILES['file_path']
        except Exception:
            return Response({"error": "file_path is required"}, status=400)

        doc_type = request.data.get("document_type", "sop")
        doc_name = request.data.get("document_name") or f.name

        s3_key = f'documents/{user.id}/{uuid.uuid4()}_{f.name}'
        s3_url = upload_file_to_s3(f, s3_key)

        ref = ReferenceDocument.objects.create(
            name=doc_name,
            document_type=doc_type,
            file_path=s3_url,
            original_filename=f.name,
            file_size=f.size,
            content_type=getattr(f, "content_type", None) or "application/octet-stream",
            uploaded_by=user,
            upload_status="processing",
        )

        # Extract text for local features/reporting
        text_content = extract_text_from_s3(s3_url)
        ref.extracted_text = text_content
        ref.upload_status = 'processed'
        ref.save()
        AuditLog.objects.create(
            action='document_upload',
            user=user,
            object_id=str(ref.id),
            object_type='ReferenceDocument',
            details={
                'document_type': ref.document_type,
                'original_filename': ref.original_filename,
                'file_size': ref.file_size,
                'source': 'UploadReferenceDocumentView',
            },
        )

        # ---- RAG ingestion (aligned to RAGitify payloads) ----
        if rag_feature_enabled():
            token_val, vs_id, err_vs = ensure_user_vector_store_id(user)
            if token_val and vs_id:
                try:
                    # set the user’s VS on this document if missing
                    if not ref.rag_vector_store_id:
                        ref.rag_enabled = True
                        ref.rag_vector_store_id = str(vs_id)
                        ref.save(update_fields=["rag_enabled", "rag_vector_store_id"])

                    # Ingest via S3 URL (no 'text', 'filename', or 'metadata' — not supported)
                    if ref.rag_vector_store_id and not ref.rag_document_id:
                        ingest = document_ingest(
                            token=token_val,
                            vector_store_id=ref.rag_vector_store_id,
                            # file=f,   # <-- correct field
                            s3_file_url=ref.file_path,  # <- use URL, not the file object
                        )
                        print("Document ingest response:", ingest)
                        doc_id = (ingest or {}).get("id") or (ingest or {}).get("document_id")
                        status_str = (ingest or {}).get("status") or "queued"

                        if doc_id:
                            ref.rag_document_id = str(doc_id)
                            ref.rag_status = status_str
                            ref.rag_uploaded_at = timezone.now()
                            ref.rag_last_error = None
                            ref.rag_metadata = ingest or {}
                            ref.save(update_fields=[
                                "rag_document_id", "rag_status",
                                "rag_uploaded_at", "rag_last_error", "rag_metadata"
                            ])

                            # Optional: quick status check
                            try:
                                st = document_status(token_val, document_id=str(doc_id)) or {}
                                s = st.get("status") or st.get("state")
                                if s:
                                    ref.rag_status = s
                                if (s or "").lower() in ("completed", "ready", "processed"):
                                    ref.rag_ingested_at = timezone.now()
                                if st.get("error"):
                                    ref.rag_last_error = st.get("error")
                                ref.rag_metadata = {**(ref.rag_metadata or {}), "status_check": st}
                                ref.save(update_fields=["rag_status", "rag_ingested_at", "rag_last_error", "rag_metadata"])
                            except Exception:
                                # Best-effort — ignore errors from the status poll
                                pass
                except Exception as e:
                    ref.rag_enabled = True
                    ref.rag_last_error = str(e)
                    ref.save(update_fields=["rag_enabled", "rag_last_error"])

        return Response(
            {"message": "Document Upload Success", "reference_document_id": str(ref.id)},
            status=200
        )

class ReferenceDocumentDetailView(GenericAPIView):
    serializer_class = ReferenceDocumentDetailSerializer

    def _get_document_for_user(self, document_id, user):
        queryset = ReferenceDocument.objects.filter(id=document_id)
        if getattr(user, 'role', None) != 'admin' and not getattr(user, 'is_staff', False):
            queryset = queryset.filter(uploaded_by=user)
        document = queryset.first()
        if not document:
            raise Http404("Reference document not found")
        return document

    def get(self, request, token, document_id, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        document = self._get_document_for_user(document_id, auth_result['user'])
        serializer = self.serializer_class(document)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, token, document_id, *args, **kwargs):
        return self._update(request, token, document_id, partial=False)

    def patch(self, request, token, document_id, *args, **kwargs):
        return self._update(request, token, document_id, partial=True)

    def _update(self, request, token, document_id, partial):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        user = auth_result['user']
        document = self._get_document_for_user(document_id, user)

        serializer = ReferenceDocumentUpdateSerializer(document, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)

        original = {
            'name': document.name,
            'document_type': document.document_type,
        }
        validated = serializer.validated_data
        changes = {}
        for field, new_value in validated.items():
            if new_value != original.get(field):
                changes[field] = {'from': original.get(field), 'to': new_value}

        serializer.save()
        document.refresh_from_db()

        AuditLog.objects.create(
            action='document_update',
            user=user,
            object_id=str(document.id),
            object_type='ReferenceDocument',
            details={
                'changes': changes,
                'source': 'ReferenceDocumentDetailView',
            },
        )

        detail_serializer = ReferenceDocumentDetailSerializer(document)
        return Response(detail_serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, token, document_id, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        user = auth_result['user']
        document = self._get_document_for_user(document_id, user)

        doc_id = str(document.id)
        file_url = document.file_path
        deleted_from_s3 = False
        s3_error = None

        if file_url:
            try:
                key = get_s3_key_from_url(file_url)
                if key:
                    s3_client.delete_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=key)
                    deleted_from_s3 = True
            except Exception as exc:
                logger.exception("Failed to delete reference document %s from S3", doc_id)
                s3_error = str(exc)

        name = document.name
        document_type = document.document_type
        document.delete()

        details = {
            'name': name,
            'document_type': document_type,
            'deleted_from_s3': deleted_from_s3,
            'source': 'ReferenceDocumentDetailView',
        }
        if s3_error:
            details['s3_error'] = s3_error

        AuditLog.objects.create(
            action='document_delete',
            user=user,
            object_id=doc_id,
            object_type='ReferenceDocument',
            details=details,
        )

        return Response(status=status.HTTP_204_NO_CONTENT)

class AudioFileDetailView(GenericAPIView):
    serializer_class = AudioFileFullSerializer

    def _get_audio_for_user(self, audio_id, user):
        queryset = AudioFile.objects.filter(id=audio_id)
        if getattr(user, 'role', None) != 'admin' and not getattr(user, 'is_staff', False):
            queryset = queryset.filter(user=user)
        audio = queryset.first()
        if not audio:
            raise Http404("Audio file not found")
        return audio

    def _get_reference_for_user(self, reference_id, user):
        queryset = ReferenceDocument.objects.filter(id=reference_id)
        if getattr(user, 'role', None) != 'admin' and not getattr(user, 'is_staff', False):
            queryset = queryset.filter(uploaded_by=user)
        reference = queryset.first()
        if not reference:
            raise Http404("Reference document not found")
        return reference

    def get(self, request, token, audio_id, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        audio = self._get_audio_for_user(audio_id, auth_result['user'])
        serializer = self.serializer_class(audio)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, token, audio_id, *args, **kwargs):
        return self._update(request, token, audio_id)

    def patch(self, request, token, audio_id, *args, **kwargs):
        return self._update(request, token, audio_id)

    def _update(self, request, token, audio_id):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        user = auth_result['user']
        audio = self._get_audio_for_user(audio_id, user)

        serializer = AudioFileUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        original_state = {
            'original_filename': audio.original_filename,
            'status': audio.status,
            'diarization_status': audio.diarization_status,
            'summary': audio.summary,
            'keywords_detected': audio.keywords_detected,
            'reference_document_id': str(audio.reference_document_id) if audio.reference_document_id else None,
        }

        validated = serializer.validated_data

        if 'original_filename' in validated:
            audio.original_filename = validated['original_filename']

        if 'status' in validated:
            audio.status = validated['status']

        if 'diarization_status' in validated:
            audio.diarization_status = validated['diarization_status']

        if 'summary' in validated:
            audio.summary = validated['summary']

        if 'keywords_detected' in validated:
            keywords = validated['keywords_detected'] or []
            audio.keywords_detected = json.dumps(keywords)

        if 'reference_document_id' in validated:
            reference_id = validated['reference_document_id']
            if reference_id:
                reference = self._get_reference_for_user(reference_id, user)
                audio.reference_document = reference
            else:
                audio.reference_document = None

        audio.save()
        audio.refresh_from_db()

        updated_state = {
            'original_filename': audio.original_filename,
            'status': audio.status,
            'diarization_status': audio.diarization_status,
            'summary': audio.summary,
            'keywords_detected': audio.keywords_detected,
            'reference_document_id': str(audio.reference_document_id) if audio.reference_document_id else None,
        }

        changes = {}
        for field, original_value in original_state.items():
            if updated_state.get(field) != original_value:
                changes[field] = {'from': original_value, 'to': updated_state.get(field)}

        AuditLog.objects.create(
            action='audiofile_update',
            user=user,
            object_id=str(audio.id),
            object_type='AudioFile',
            details={
                'changes': changes,
                'source': 'AudioFileDetailView',
            },
        )

        response_serializer = self.serializer_class(audio)
        return Response(response_serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, token, audio_id, *args, **kwargs):
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            return Response({'error': auth_result['error']}, status=status.HTTP_401_UNAUTHORIZED)

        user = auth_result['user']
        audio = self._get_audio_for_user(audio_id, user)

        audio_id_str = str(audio.id)
        file_url = audio.file_path
        deleted_from_s3 = False
        s3_error = None

        if file_url:
            try:
                key = get_s3_key_from_url(file_url)
                if key:
                    s3_client.delete_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=key)
                    deleted_from_s3 = True
            except Exception as exc:
                logger.exception("Failed to delete audio file %s from S3", audio_id_str)
                s3_error = str(exc)

        original_filename = audio.original_filename
        audio.delete()

        details = {
            'original_filename': original_filename,
            'file_path': file_url,
            'deleted_from_s3': deleted_from_s3,
            'source': 'AudioFileDetailView',
        }
        if s3_error:
            details['s3_error'] = s3_error

        AuditLog.objects.create(
            action='audiofile_delete',
            user=user,
            object_id=audio_id_str,
            object_type='AudioFile',
            details=details,
        )

        return Response(status=status.HTTP_204_NO_CONTENT)

# -------------------------- RAG Conversational APIs --------------------------

class RAGAssistantCreateView(CreateAPIView):
    serializer_class = RAGAssistantSerializer
    @swagger_auto_schema(operation_description="Create RAG Assistant bound to vector_store_ids")
    def post(self, request, token):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error'], 'timestamp': timezone.now()}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)

        user: UserProfile = auth['user']
        rag_token, err = ensure_rag_token(user)
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)

        name = request.data.get("name") or "3PC Assistant"
        vector_store_ids = request.data.get("vector_store_ids") or []
        model = request.data.get("model") or "gpt-4o"

        try:
            resp = assistant_create(rag_token, name=name, vector_store_ids=vector_store_ids, model=model)
            asst_id = (resp or {}).get("id") or (resp or {}).get("assistant_id")
            if asst_id:
                RAGAssistant.objects.get_or_create(
                    external_id=asst_id,
                    defaults={
                        "name": name, "model": model,
                        "vector_store_ids": vector_store_ids, "owner": user
                    }
                )
            return Response(resp, status=201)
        except Exception as e:
            logger.exception("assistant_create failed")
            return Response({'error': str(e)}, status=500)

class RAGAssistantListView(APIView):
    def get(self, request, token):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)
        try:
            return Response(assistant_list(rag_token), status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGAssistantDetailView(APIView):
    def get(self, request, token, assistant_id):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)
        try:
            return Response(assistant_detail(rag_token, assistant_id), status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGThreadCreateView(CreateAPIView):
    serializer_class = RAGThreadSerializer
    def post(self, request, token):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        user = auth['user']
        rag_token, err = ensure_rag_token(user)
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)

        assistant_id = request.data.get("assistant_id")
        title = request.data.get("title")
        try:
            resp = thread_create(rag_token, assistant_id=assistant_id, title=title)
            thread_id = (resp or {}).get("id") or (resp or {}).get("thread_id")
            if thread_id:
                RAGThread.objects.get_or_create(
                    external_id=thread_id,
                    defaults={"assistant_external_id": assistant_id, "title": title, "owner": user}
                )
            return Response(resp, status=201)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGThreadListView(APIView):
    def get(self, request, token):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)
        try:
            return Response(thread_list(rag_token), status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGThreadDetailView(APIView):
    def get(self, request, token, thread_id):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)
        try:
            return Response(thread_detail(rag_token, thread_id), status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGThreadMessagesView(APIView):
    def get(self, request, token, thread_id):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)
        try:
            msgs = thread_messages(rag_token, thread_id)
            # persist locally (best-effort)
            for m in msgs or []:
                mid = m.get("id") or m.get("message_id")
                if not mid: continue
                RAGMessage.objects.get_or_create(
                    external_id=mid,
                    defaults={
                        "thread_external_id": thread_id,
                        "role": m.get("role", "assistant"),
                        "content": m.get("content", ""),
                    }
                )
            return Response(msgs, status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGMessageCreateView(CreateAPIView):
    serializer_class = RAGMessageSerializer
    def post(self, request, token):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)

        thread_id = request.data.get("thread_id")
        content = request.data.get("content")
        role = request.data.get("role", "user")
        try:
            resp = message_create(rag_token, thread_id=thread_id, content=content, role=role)
            mid = (resp or {}).get("id") or (resp or {}).get("message_id")
            if mid:
                RAGMessage.objects.get_or_create(
                    external_id=mid,
                    defaults={"thread_external_id": thread_id, "role": role, "content": content}
                )
            return Response(resp, status=201)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGMessageListView(APIView):
    def get(self, request, token):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)
        try:
            return Response(message_list(rag_token), status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGMessageDetailView(APIView):
    def get(self, request, token, message_id):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)
        try:
            return Response(message_detail(rag_token, message_id), status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGRunCreateView(CreateAPIView):
    serializer_class = RAGRunSerializer
    def post(self, request, token):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        user = auth['user']
        rag_token, err = ensure_rag_token(user)
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)

        thread_id = request.data.get("thread_id")
        assistant_id = request.data.get("assistant_id")
        try:
            resp = run_create(rag_token, thread_id=thread_id, assistant_id=assistant_id)
            rid = (resp or {}).get("id") or (resp or {}).get("run_id")
            if rid:
                RAGRun.objects.get_or_create(
                    external_id=rid,
                    defaults={
                        "thread_external_id": thread_id,
                        "assistant_external_id": assistant_id,
                        "status": (resp or {}).get("status"),
                        "raw": resp or {}
                    }
                )
            return Response(resp, status=201)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGRunListView(APIView):
    def get(self, request, token):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)
        try:
            return Response(run_list(rag_token), status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGRunDetailView(APIView):
    def get(self, request, token, run_id):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)
        try:
            return Response(run_detail(rag_token, run_id), status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class RAGRunSubmitToolOutputsView(CreateAPIView):
    def post(self, request, token, run_id):
        auth = token_verification(token)
        if auth['status'] != 200:
            return Response({'error': auth['error']}, status=401)
        if not rag_feature_enabled():
            return Response({'error': 'RAG disabled'}, status=404)
        rag_token, err = ensure_rag_token(auth['user'])
        if not rag_token:
            return Response({'error': f'No RAG token: {err}'}, status=500)
        tool_outputs = request.data.get("tool_outputs") or []
        try:
            return Response(run_submit_tool_outputs(rag_token, run_id, tool_outputs), status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)
