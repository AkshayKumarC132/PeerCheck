import os
import uuid
import boto3
from rest_framework.views import APIView
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework import status
from .models import AudioFile, SOP, SOPStep, Session, SessionUser, UserSettings, SystemSettings, AuditLog
from .serializers import (AudioFileSerializer, FeedbackSerializer, ProcessAudioViewSerializer, 
        SOPSerializer, SessionSerializer,FeedbackReviewSerializer, UserSettingsSerializer, SystemSettingsSerializer, AuditLogSerializer)
from .utils import *
from peercheck import settings
from fuzzywuzzy import fuzz
from Levenshtein import distance
from .authentication import token_verification
from .permissions import RoleBasedPermission
import logging

logger = logging.getLogger(__name__)

try:
    # AWS S3 Configuration
    S3_BUCKET_NAME = settings.AWS_STORAGE_BUCKET_NAME
    S3_REGION = settings.AWS_S3_REGION_NAME
    S3_ACCESS_KEY = settings.AWS_S3_ACCESS_KEY_ID
    S3_SECRET_KEY = settings.AWS_S3_SECRET_ACCESS_KEY

    # Initialize S3 client
    s3_client = boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {str(e)}")
    pass

MODEL_PATH = os.path.join(settings.BASE_DIR, "vosk-model-small-en-us-0.15")

SPEAKER_MODEL_PATH = os.path.join(settings.BASE_DIR, "vosk-model-spk-0.4")

def upload_file_to_s3(file, bucket_name, file_name):
    """
    Upload a file to an S3 bucket.
    """
    try:
        s3_client.upload_fileobj(file, bucket_name, file_name)
        return f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
    except Exception as e:
        raise Exception(f"Error uploading file to S3: {str(e)}")


def download_file_from_s3(file_name, bucket_name):
    """
    Download a file from an S3 bucket.
    """
    try:
        file_path = f"/uploads/{uuid.uuid4()}_{os.path.basename(file_name)}"
        s3_client.download_file(bucket_name, file_name, file_path)
        return file_path
    except Exception as e:
        raise Exception(f"Error downloading file from S3: {str(e)}")


class ProcessAudioView(CreateAPIView):
    
    serializer_class = ProcessAudioViewSerializer
    permission_classes = [RoleBasedPermission]

    def post(self, request, token,format=None):
        # request.validated_user is set by RoleBasedPermission
        # No need for token_verification call here as RoleBasedPermission handles it.
        print(f"User data (validated_user): {request.validated_user.username if request.validated_user else 'None'}"
              f"Request data: {request.data}")
        serializer = ProcessAudioViewSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        audio_file = serializer.validated_data.get("file")
        # start_prompt = request.data.get("start_prompt")
        # end_prompt = request.data.get("end_prompt")
        keywords = request.data.get("keywords", "")
        sop_id = request.data.get("sop_id") 
        session_id = serializer.validated_data.get("session_id")
        session_user_ids = serializer.validated_data.get("session_user_ids", [])

        # if not audio_file or not start_prompt or not end_prompt:
        #     return Response({"error": "Missing Start or End Prompt fields."}, status=status.HTTP_400_BAD_REQUEST)

        file_name = f"peercheck_files/{uuid.uuid4()}_{audio_file.name}"
        try:
            file_url = upload_file_to_s3(audio_file, S3_BUCKET_NAME, file_name)
        except Exception as e:
            logger.error(f"S3 upload failed: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        session_obj = None
        if session_id:
            try:
                # Fetch session; ownership/permission check might be deferred to RoleBasedPermission or done here
                session_obj = Session.objects.get(id=session_id)
            except Session.DoesNotExist:
                logger.error(f"Session not found: ID {session_id}")
                return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
            except Exception as e: # General exception
                logger.error(f"Error fetching session {session_id}: {str(e)}")
                return Response({"error": f"Error fetching session: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Placeholder for transcription result
        transcription_segments = []
        transcription_text = ""
        try:
            # Call to actual or placeholder function
            transcription_segments = transcribe_with_speaker_diarization(file_url, MODEL_PATH, SPEAKER_MODEL_PATH, session_id)
            if not isinstance(transcription_segments, list) or not all(isinstance(seg, dict) and 'text' in seg and 'speaker' in seg for seg in transcription_segments):
                 logger.error(f"Transcription result is not in the expected format: {transcription_segments}")
                 return Response({"error": "Transcription service returned unexpected data format."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            transcription_text = " ".join([segment["text"] for segment in transcription_segments])
            logger.info(f"Transcription segments: {transcription_segments}")
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return Response({"error": f"Transcription failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        audio_instance = AudioFile.objects.create(
            file_path=file_url,
            transcription=transcription_segments, # Store structured transcription
            status="processed",
            duration=len(transcription_text.split()), # Approximate duration
            session=session_obj, # Link to session object
            # speaker_tag for AudioFile itself could be primary speaker, or null if multiple.
            # This depends on how `speaker_tag` on AudioFile model is intended to be used.
            # For now, let's assume it's for a primary speaker, or can be derived later.
        )
        logger.info(f"Created AudioFile instance: {audio_instance.id}")

        # Speaker Diarization & SessionUser:
        if session_obj and session_user_ids:
            unique_speaker_tags = sorted(list(set(seg['speaker'] for seg in transcription_segments)))
            for i, speaker_tag_from_transcription in enumerate(unique_speaker_tags):
                if i < len(session_user_ids):
                    user_id_for_speaker = session_user_ids[i]
                    try:
                        user_profile_for_speaker = UserProfile.objects.get(id=user_id_for_speaker)
                        su, created = SessionUser.objects.get_or_create(
                            session=session_obj,
                            user=user_profile_for_speaker,
                            defaults={'speaker_tag': speaker_tag_from_transcription}
                        )
                        if not created and su.speaker_tag != speaker_tag_from_transcription:
                            su.speaker_tag = speaker_tag_from_transcription
                            su.save()
                        logger.info(f"Associated/Updated SessionUser: user_id={user_id_for_speaker} with speaker_tag={speaker_tag_from_transcription} for session_id={session_obj.id}")
                    except UserProfile.DoesNotExist:
                        logger.warning(f"UserProfile with id {user_id_for_speaker} not found for speaker tag assignment for session {session_obj.id}.")
                else:
                    logger.warning(f"Not enough user_ids in session_user_ids to map all unique_speaker_tags for session {session_obj.id}. Speaker_tag '{speaker_tag_from_transcription}' unassigned.")

        AuditLog.objects.create(
            action='audio_upload',
            user=request.validated_user,
            session=session_obj, # Use the session object
            object_id=audio_instance.id,
            object_type='AudioFile',
            details={'file_name': audio_file.name}
        )

        response_data = {
            "transcription": transcription_segments, # Return structured transcription
            "status": "processed",
            "audio_file": AudioFileSerializer(audio_instance).data
        }

        # SOP Matching
        sop_to_use = None
        if sop_id:
            try:
                sop_to_use = SOP.objects.get(id=sop_id)
            except SOP.DoesNotExist:
                logger.warning(f"SOP with id {sop_id} from request not found.")
                response_data["sop_error"] = f"SOP with id {sop_id} not found."
        elif session_obj and session_obj.sop:
            sop_to_use = session_obj.sop

        if sop_to_use:
            try:
                # match_sop_steps needs to be robust.
                # Assuming it takes combined text and SOP object (or its steps)
                sop_matches = match_sop_steps(transcription_text, sop_to_use)
                response_data["sop_matches"] = sop_matches
                logger.info(f"SOP matching completed for SOP ID {sop_to_use.id}")
            except Exception as e:
                logger.error(f"SOP matching failed for SOP {sop_to_use.id}: {str(e)}")
                response_data["sop_error"] = f"SOP matching failed: {str(e)}"
        
        # AudioFile is already linked to session_obj upon creation.

        logger.info("Audio processing completed successfully")
        return Response(response_data, status=status.HTTP_200_OK)


class FeedbackView(APIView):
    permission_classes = [RoleBasedPermission]

    def post(self, request, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        serializer = FeedbackSerializer(data=request.data)
        if serializer.is_valid():
            # The Feedback model does not have a direct user link (e.g., submitted_by)
            # If it did, it would be: feedback = serializer.save(submitted_by=request.validated_user)
            feedback = serializer.save()

            session_for_audit = None
            if feedback.audio_file and feedback.audio_file.session:
                session_for_audit = feedback.audio_file.session

            AuditLog.objects.create(
                action='feedback_submit',
                user=request.validated_user,
                session=session_for_audit,
                object_id=feedback.id,
                object_type='Feedback',
                details={'audio_file_id': feedback.audio_file.id}
            )
            return Response({"message": "Feedback submitted successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class GetAudioRecordsView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        user = request.validated_user
        try:
            if user.role == 'admin':
                audio_records = AudioFile.objects.all()
            elif user.role == 'operator':
                # AudioFiles linked to sessions created by the operator
                audio_records = AudioFile.objects.filter(session__created_by=user)
            elif user.role == 'reviewer':
                # Reviewers see AudioFiles from sessions they are part of or sessions 'under_review'
                audio_records = AudioFile.objects.filter(
                    Q(session__session_users__user=user) | Q(session__status='under_review')
                ).distinct()
            else:
                audio_records = AudioFile.objects.none()

            audio_records = audio_records.order_by("-id")
            serializer = AudioFileSerializer(audio_records, many=True)
            return Response({"audio_records": serializer.data}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Error fetching audio records: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ReAnalyzeAudioView(APIView):
    permission_classes = [RoleBasedPermission]

    def post(self, request, token):
        # request.validated_user is set by RoleBasedPermission
        user = request.validated_user

        file_path = request.data.get("file_path")
        new_keywords = request.data.get("new_keywords", "") # Optional for re-analysis

        if not file_path:
            return Response({"error": "File path is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            audio_instance = AudioFile.objects.get(file_path=file_path)
        except AudioFile.DoesNotExist:
            return Response({"error": "AudioFile not found"}, status=status.HTTP_404_NOT_FOUND)

        # Re-run transcription and diarization (placeholder returns dummy data)
        try:
            transcription_segments = transcribe_with_speaker_diarization(audio_instance.file_path, MODEL_PATH, SPEAKER_MODEL_PATH, audio_instance.session_id)
            if not isinstance(transcription_segments, list) or not all(isinstance(seg, dict) and 'text' in seg and 'speaker' in seg for seg in transcription_segments):
                 logger.error(f"Re-transcription result is not in the expected format: {transcription_segments}")
                 return Response({"error": "Re-transcription service returned unexpected data format."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            transcription_text = " ".join([segment["text"] for segment in transcription_segments])
        except Exception as e:
            logger.error(f"Re-transcription failed for {audio_instance.file_path}: {str(e)}")
            return Response({"error": f"Re-transcription failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        keywords_detected_str = audio_instance.keywords_detected # Default to existing
        if new_keywords: # Only re-detect if new_keywords are provided
            keywords_list = [kw.strip() for kw in new_keywords.split(",")]
            detected_keywords_obj = detect_keywords(transcription_text, keywords_list)
            # Store as JSON string or appropriate format based on detect_keywords output
            keywords_detected_str = json.dumps(detected_keywords_obj) if isinstance(detected_keywords_obj, (dict, list)) else str(detected_keywords_obj)

        audio_instance.transcription = transcription_segments
        audio_instance.keywords_detected = keywords_detected_str
        audio_instance.status = "reanalyzed"
        audio_instance.duration = len(transcription_text.split()) # Update duration
        audio_instance.save()
        
        AuditLog.objects.create(
            action='audio_reanalyze', 
            user=user,
            session=audio_instance.session,
            object_id=audio_instance.id,
            object_type='AudioFile',
            details={'file_path': audio_instance.file_path, 'new_keywords_provided': bool(new_keywords)}
        )

        return Response(
            {
                "file_path": audio_instance.file_path,
                "transcription": audio_instance.transcription,
                "detected_keywords": audio_instance.keywords_detected,
                "status": audio_instance.status,
                "audio_file": AudioFileSerializer(audio_instance).data
            },
            status=status.HTTP_200_OK,
        )

def find_prompt_index(transcription, prompt, threshold=80):
    """
    Find the position of a prompt in transcription with fuzzy matching.
    
    :param transcription: Transcribed text from audio.
    :param prompt: The start or end prompt provided by the user.
    :param threshold: Matching threshold percentage.
    :return: Index of the best match if found, else -1.
    """
    words = transcription.lower().split()
    prompt_words = prompt.lower().split()
    best_match_index = -1

    for i in range(len(words) - len(prompt_words) + 1):
        segment = " ".join(words[i:i+len(prompt_words)])
        similarity = fuzz.ratio(segment, prompt.lower())

        if similarity >= threshold:
            best_match_index = transcription.lower().find(segment)
            break

    return best_match_index

def find_approximate_match(transcription, prompt):
    words = transcription.lower().split()
    prompt_words = prompt.lower().split()
    min_distance = float('inf')
    best_match_index = -1

    for i in range(len(words) - len(prompt_words) + 1):
        segment = " ".join(words[i:i+len(prompt_words)])
        current_distance = distance(segment, prompt.lower())

        if current_distance < min_distance:
            min_distance = current_distance
            best_match_index = transcription.lower().find(segment)

    return best_match_index if min_distance < len(prompt) * 0.3 else -1



class SOPCreateView(CreateAPIView):
    serializer_class = SOPSerializer
    permission_classes = [RoleBasedPermission]

    def post(self, request, token):
        # request.validated_user is set by RoleBasedPermission
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            sop = serializer.save(created_by=request.validated_user)
            AuditLog.objects.create(
                action='sop_create',
                user=request.validated_user,
                object_id=sop.id,
                object_type='SOP',
                details={'name': sop.name, 'version': sop.version}
            )
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SOPListView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token):
        # request.validated_user is set by RoleBasedPermission
        # SOPs are generally public for all authenticated users as per requirements
        sops = SOP.objects.all().order_by('name', '-version')
        serializer = SOPSerializer(sops, many=True)
        return Response({'sops': serializer.data}, status=status.HTTP_200_OK)
    
class SessionCreateView(CreateAPIView):
    serializer_class = SessionSerializer
    permission_classes = [RoleBasedPermission]

    def post(self, request, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        serializer = self.serializer_class(data=request.data, context={'request': request})
        if serializer.is_valid():
            # Session model's 'created_by' is now set using request.validated_user
            session = serializer.save(created_by=request.validated_user)

            sop_id = request.data.get('sop_id')
            if sop_id:
                try:
                    sop = SOP.objects.get(id=sop_id)
                    session.sop = sop
                    session.save(update_fields=['sop'])
                except SOP.DoesNotExist:
                    logger.warning(f"SOP with id {sop_id} not found during session creation for session {session.id}")
            
            session_user_ids = request.data.get('session_user_ids', [])
            if isinstance(session_user_ids, str): # Handle if passed as JSON string from form-data
                try:
                    session_user_ids = json.loads(session_user_ids)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON format for session_user_ids: {session_user_ids}")
                    session_user_ids = []
            
            if not isinstance(session_user_ids, list):
                logger.warning(f"session_user_ids is not a list: {session_user_ids}")
                session_user_ids = []

            # Add creator to SessionUser by default
            SessionUser.objects.get_or_create(session=session, user=request.validated_user)

            for user_id in session_user_ids:
                try:
                    user_to_add = UserProfile.objects.get(id=user_id)
                    SessionUser.objects.get_or_create(session=session, user=user_to_add)
                except UserProfile.DoesNotExist:
                    logger.warning(f"User with id {user_id} not found for adding to session {session.id}")
            
            AuditLog.objects.create(
                action='session_create',
                user=request.validated_user,
                session=session,
                object_id=session.id,
                object_type='Session',
                details={'name': session.name}
            )
            # Return the serialized session data, including the now correctly populated 'created_by'
            return Response(SessionSerializer(session).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SessionListView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        user = request.validated_user
        try:
            if user.role == 'admin':
                sessions = Session.objects.all()
            elif user.role == 'operator':
                # Sessions created by the operator
                sessions = Session.objects.filter(created_by=user)
            elif user.role == 'reviewer':
                # Reviewers see sessions they are part of (SessionUser) OR sessions marked 'under_review'
                sessions = Session.objects.filter(
                    Q(session_users__user=user) | Q(status='under_review')
                ).distinct()
            else:
                sessions = Session.objects.none()
                
            sessions = sessions.order_by("-created_at")
            serializer = SessionSerializer(sessions, many=True)
            return Response({'sessions': serializer.data}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching sessions: {str(e)}")
            return Response({"error": f"Error fetching sessions: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class SessionReviewView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, session_id, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        logger.info(f"Fetching review data for session ID: {session_id}, user: {request.validated_user.username if request.validated_user else 'None'}")
        try:
            session = Session.objects.prefetch_related(
                'audio_files', 'sop__steps', 
                'feedback_reviews__reviewer', 'session_users__user' # Ensure related names are correct
            ).get(id=session_id)
            
            # Add object-level permission check if RoleBasedPermission doesn't cover it for this view
            # e.g., if request.validated_user.role == 'reviewer' and not (session.status == 'under_review' or session.session_users.filter(user=request.validated_user).exists()):
            #    return Response({"error": "Forbidden to review this session"}, status=status.HTTP_403_FORBIDDEN)

            serializer = SessionSerializer(session)
            logger.info(f"Retrieved review data for session: {session.name}")
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Session.DoesNotExist:
            logger.error(f"Session not found: ID {session_id}")
            return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error fetching session review for session {session_id}: {str(e)}")
            return Response({"error": f"Error fetching session review: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request, session_id, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        logger.info(f"Submitting review for session ID: {session_id}, user: {request.validated_user.username if request.validated_user else 'None'}")
        try:
            session = Session.objects.get(id=session_id)
            data = request.data.copy()
            # FeedbackReviewSerializer expects 'reviewer_id' and 'session' (PK)
            data['reviewer_id'] = request.validated_user.id
            data['session'] = session.id 
            
            serializer = FeedbackReviewSerializer(data=data)
            if serializer.is_valid():
                feedback_review = serializer.save() # Serializer handles linking reviewer_id to reviewer object
                
                AuditLog.objects.create(
                    action='review_submit',
                    user=request.validated_user,
                    session=session,
                    object_id=feedback_review.id,
                    object_type='FeedbackReview',
                    details={'comments_preview': feedback_review.comments[:50]}
                )
                logger.info(f"Feedback review submitted successfully for session {session_id}")
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            logger.error(f"Feedback review submission failed for session {session_id}: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Session.DoesNotExist:
            logger.error(f"Session not found for review submission: ID {session_id}")
            return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error submitting feedback review for session {session_id}: {str(e)}")
            return Response({"error": f"Error submitting feedback review: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

import wave
import json
import tempfile
import requests # Ensure requests is imported
import vosk # Ensure vosk is imported
from pydub import AudioSegment # For audio conversion

# Actual implementation of the transcription and diarization function
def transcribe_with_speaker_diarization(audio_url: str, vosk_model_path: str, speaker_model_path: str, session_id: int = None):
    logger.info(f"Starting transcription and diarization for audio URL: {audio_url}")
    
    temp_audio_path = None
    converted_audio_path = None

    try:
        # 1. Download Audio
        response = requests.get(audio_url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Create a temporary file to save the downloaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".download") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            temp_audio_path = tmp_file.name
        logger.info(f"Audio downloaded to temporary file: {temp_audio_path}")

        # 2. Audio Format Conversion (to WAV, 16kHz, mono)
        try:
            audio = AudioSegment.from_file(temp_audio_path)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            
            # Export to a new temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as converted_tmp_file:
                audio.export(converted_tmp_file.name, format="wav")
                converted_audio_path = converted_tmp_file.name
            logger.info(f"Audio converted and saved to: {converted_audio_path}")
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            # Fallback: try to process the original if pydub fails, assuming it might be WAV already
            # This is risky, Vosk is sensitive to format.
            if temp_audio_path:
                 logger.warning(f"Attempting to use original downloaded audio {temp_audio_path} due to conversion error.")
                 converted_audio_path = temp_audio_path # Use original if conversion fails
            else:
                raise # Re-raise if original download also failed

        # 3. Vosk Transcription & Diarization
        if not os.path.exists(vosk_model_path):
            logger.error(f"Vosk ASR model path does not exist: {vosk_model_path}")
            raise FileNotFoundError(f"Vosk ASR model not found at {vosk_model_path}")
        if not os.path.exists(speaker_model_path):
            logger.error(f"Vosk Speaker model path does not exist: {speaker_model_path}")
            raise FileNotFoundError(f"Vosk Speaker model not found at {speaker_model_path}")

        vosk.SetLogLevel(-1) # Suppress Vosk logs, can be set to 0 for more detail
        model = vosk.Model(vosk_model_path)
        spk_model = vosk.SpkModel(speaker_model_path)
        
        final_transcription_segments = []
        current_speaker = None
        current_text = []

        with wave.open(converted_audio_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                logger.error(f"Audio file {converted_audio_path} is not mono WAV with 16-bit samples. Channels: {wf.getnchannels()}, SampleWidth: {wf.getsampwidth()}, CompType: {wf.getcomptype()}")
                raise ValueError("Audio file must be WAV format mono PCM.")
            
            recognizer = vosk.KaldiRecognizer(model, wf.getframerate())
            recognizer.SetSpkModel(spk_model)
            recognizer.SetWords(True) # Enable word-level details

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if 'result' in result: # Word results
                        for word_info in result['result']:
                            # Speaker change detection based on 'spk' field (vector of confidences)
                            # A simple heuristic: if spk field significantly changes or a new one appears.
                            # Vosk's speaker diarization assigns a speaker vector; comparing these is complex.
                            # A more practical approach is to look at 'conf' and speaker tag changes in partial results.
                            # For now, we rely on the 'spk' field in the final partial result of a segment.
                            # This example processes word by word, which is more complex for speaker change.
                            # A simpler method is to process segments.
                            pass # Word processing logic would be here if needed for finer detail

            # Process final result to get segments
            final_result_json = recognizer.FinalResult()
            final_result = json.loads(final_result_json)
            
            # The 'spk' field in final_result might not be directly usable for segmenting the whole text.
            # Instead, we iterate through 'result' for word-level speaker info if available
            # and reconstruct segments.
            
            if 'result' in final_result and final_result['result']: # Check if word details are present
                for word_info in final_result['result']:
                    speaker_tag_id = None
                    # Check for 'spk' key and if it's a list with elements
                    if 'spk' in word_info and isinstance(word_info['spk'], list) and word_info['spk']:
                        # Heuristic: take the speaker with highest confidence from the spk vector
                        # This is a simplification; proper spk vector comparison is needed for robust diarization
                        speaker_tag_id = word_info['spk'].index(max(word_info['spk']))
                    elif 'speaker' in word_info : # Some vosk versions might use 'speaker' directly at word level
                         speaker_tag_id = word_info['speaker']


                    speaker_label = f"Speaker_{int(speaker_tag_id) + 1}" if speaker_tag_id is not None else "Unknown"

                    if current_speaker is None:
                        current_speaker = speaker_label
                    
                    if speaker_label != current_speaker and current_text:
                        final_transcription_segments.append({
                            'speaker': current_speaker,
                            'text': " ".join(current_text).strip()
                        })
                        current_text = []
                        current_speaker = speaker_label
                    
                    current_text.append(word_info['word'])
                
                # Append any remaining text
                if current_text:
                    final_transcription_segments.append({
                        'speaker': current_speaker if current_speaker else "Unknown",
                        'text': " ".join(current_text).strip()
                    })
            elif 'text' in final_result and final_result['text']: # Fallback if no word-level speaker data
                 # If 'spk' is present at the top level of final_result, use it, otherwise "Unknown"
                speaker_tag_id = None
                if 'spk' in final_result and isinstance(final_result['spk'], list) and final_result['spk']:
                    speaker_tag_id = final_result['spk'].index(max(final_result['spk']))
                speaker_label = f"Speaker_{int(speaker_tag_id) + 1}" if speaker_tag_id is not None else "Unknown"
                final_transcription_segments.append({
                    'speaker': speaker_label,
                    'text': final_result['text'].strip()
                })


        if not final_transcription_segments: # Ensure we return something if processing was minimal
            logger.warning(f"Transcription for {audio_url} resulted in empty segments. Original text (if any): {final_result.get('text', '')}")
            # Fallback to full text with unknown speaker if segments are empty but text exists
            if 'text' in final_result and final_result['text']:
                 final_transcription_segments.append({'speaker': 'Unknown', 'text': final_result['text'].strip()})

        logger.info(f"Transcription and diarization completed. Segments: {len(final_transcription_segments)}")
        return final_transcription_segments

    except requests.exceptions.RequestException as e:
        logger.error(f"Audio download failed from {audio_url}: {e}")
        raise Exception(f"Audio download failed: {e}") from e
    except FileNotFoundError as e: # For model path errors
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during transcription/diarization for {audio_url}: {e}")
        # Fallback: return a single segment with error message or empty list
        # For robustness, one might return what has been processed so far, or specific error segment.
        # For now, re-raise to indicate failure to the caller.
        raise Exception(f"Transcription/diarization process failed: {e}") from e
    finally:
        # 5. Cleanup Temporary Files
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Cleaned up temporary downloaded file: {temp_audio_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_audio_path}: {e}")
        if converted_audio_path and converted_audio_path != temp_audio_path and os.path.exists(converted_audio_path):
            try:
                os.remove(converted_audio_path)
                logger.info(f"Cleaned up temporary converted file: {converted_audio_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temp file {converted_audio_path}: {e}")

# Comment out or remove the old placeholder if this is replacing it directly in views.py
# The old placeholder was:
# def transcribe_with_speaker_diarization(audio_url: str, model_path: str, speaker_model_path: str):
#     """
#     Transcribes audio with Vosk and includes speaker diarization.

#     Args:
#         audio_url (str): URL of the audio file.
#         model_path (str): Path to the Vosk ASR model.
#         speaker_model_path (str): Path to the Vosk speaker diarization model.

#     Returns:
#         dict: Formatted transcription with speaker labels.
#     """
#     # Download the file
#     response = requests.get(audio_url)
#     if response.status_code != 200:
#         raise ValueError(f"Failed to download file: {response.status_code}")
#     import tempfile
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#         temp_audio.write(response.content)
#         temp_audio_path = temp_audio.name

#     try:
#         model = vosk.Model(model_path)
#         spk_model = vosk.SpkModel(speaker_model_path)  # Load Speaker Model

#         with wave.open(temp_audio_path, "rb") as wf:
#             recognizer = vosk.KaldiRecognizer(model, wf.getframerate())
#             recognizer.SetWords(True)
#             recognizer.SetSpkModel(spk_model)  # Enable speaker diarization

#             transcription = []

#             while True:
#                 data = wf.readframes(4000)
#                 if len(data) == 0:
#                     break

#                 if recognizer.AcceptWaveform(data):
#                     result = json.loads(recognizer.Result())

#                     # Extract speaker ID
#                     speaker_id = None
#                     if "spk" in result:
#                         if isinstance(result["spk"], list) and result["spk"]:  # Ensure it's a non-empty list
#                             speaker_id = result["spk"][0]  # Take the first speaker ID
#                         elif isinstance(result["spk"], (int, float)):  # Handle single value
#                             speaker_id = result["spk"]

#                     # Ensure valid speaker ID
#                     speaker = f"Speaker_{int(speaker_id) + 1}" if speaker_id is not None else "Unknown"

#                     # Append transcription
#                     if "text" in result and result["text"]:
#                         transcription.append({"speaker": speaker, "text": result["text"]})

#             # Process the final result
#             final_result = json.loads(recognizer.FinalResult())
#             if "text" in final_result and final_result["text"]:
#                 speaker_id = None
#                 if "spk" in final_result:
#                     if isinstance(final_result["spk"], list) and final_result["spk"]:
#                         speaker_id = final_result["spk"][0]
#                     elif isinstance(final_result["spk"], (int, float)):
#                         speaker_id = final_result["spk"]

#                 speaker = f"Speaker_{int(speaker_id) + 1}" if speaker_id is not None else "Unknown"
#                 transcription.append({"speaker": speaker, "text": final_result["text"]})

#     finally:
#         os.remove(temp_audio_path)  # Delete temp file after processing

#     return transcription

class UserSettingsView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        logger.info(f"Fetching user settings for user: {request.validated_user.username if request.validated_user else 'None'}")
        try:
            settings_obj, created = UserSettings.objects.get_or_create(user=request.validated_user)
            serializer = UserSettingsSerializer(settings_obj)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching user settings for {request.validated_user.username if request.validated_user else 'None'}: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def patch(self, request, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        logger.info(f"Updating user settings for user: {request.validated_user.username if request.validated_user else 'None'}")
        try:
            settings_obj, created = UserSettings.objects.get_or_create(user=request.validated_user)
            serializer = UserSettingsSerializer(settings_obj, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                
                if 'theme' in request.data:
                    user_profile = request.validated_user # This is a UserProfile instance
                    user_profile.theme = request.data['theme']
                    user_profile.save(update_fields=['theme'])
                
                AuditLog.objects.create(
                    action='user_settings_update',
                    user=request.validated_user,
                    object_id=settings_obj.id, # Could be user_profile.id if settings are intrinsic part of user
                    object_type='UserSettings',
                    details={'updated_fields': list(request.data.keys())}
                )
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error updating user settings for {request.validated_user.username if request.validated_user else 'None'}: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SystemSettingsView(APIView):
    permission_classes = [RoleBasedPermission] # Ensures RoleBasedPermission checks role

    def get(self, request, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        # Role check should be handled by RoleBasedPermission class for this view.
        # If not, explicit check: if request.validated_user.role != 'admin': return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)
        logger.info(f"Fetching system settings by user: {request.validated_user.username if request.validated_user else 'None'}")
        try:
            settings_obj, created = SystemSettings.objects.get_or_create(id=1) # Assuming singleton
            serializer = SystemSettingsSerializer(settings_obj)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching system settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def patch(self, request, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        # Role check should be handled by RoleBasedPermission class for this view.
        logger.info(f"Updating system settings by user: {request.validated_user.username if request.validated_user else 'None'}")
        try:
            settings_obj, created = SystemSettings.objects.get_or_create(id=1)
            serializer = SystemSettingsSerializer(settings_obj, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                AuditLog.objects.create(
                    action='system_settings_update',
                    user=request.validated_user,
                    object_id=settings_obj.id,
                    object_type='SystemSettings',
                    details={'updated_fields': list(request.data.keys())}
                )
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error updating system settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AuditLogView(APIView):
    permission_classes = [RoleBasedPermission] # Ensures RoleBasedPermission checks role

    def get(self, request, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        # Role check should be handled by RoleBasedPermission class for this view.
        # (e.g. admin and reviewer can access)
        logger.info(f"Fetching audit logs by user: {request.validated_user.username if request.validated_user else 'None'}")
        try:
            # Future: Add filtering based on user role if reviewers should see limited logs
            logs = AuditLog.objects.all().order_by('-timestamp')
            # Consider pagination for performance with many logs
            # from django.core.paginator import Paginator
            # paginator = Paginator(logs, 25) # Show 25 logs per page
            # page_number = request.GET.get('page')
            # page_obj = paginator.get_page(page_number)
            # serializer = AuditLogSerializer(page_obj, many=True)
            # return Response({'audit_logs': serializer.data, 'num_pages': paginator.num_pages}, status=status.HTTP_200_OK)
            serializer = AuditLogSerializer(logs, many=True)
            return Response({'audit_logs': serializer.data}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching audit logs: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Required imports based on changes (ensure these are at the top of the file with other imports)
# import wave # Added within the new function block
# import json # Already present
# import tempfile # Added within the new function block
# import requests # Added within the new function block
# import vosk # Added within the new function block
# from pydub import AudioSegment # Added within the new function block
from django.contrib.auth import get_user_model
UserProfile = get_user_model()
from django.db.models import Q
# import json # Already present below, ensure it's only imported once at the top
