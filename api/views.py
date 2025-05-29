import os
import uuid
import boto3
from django.shortcuts import get_object_or_404
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
from django.db import models

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
        # Validate token
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)
        print(f"User data: {user_data}"
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

        if session_id and session_user_ids:
            try:
                session_instance = Session.objects.get(id=session_id, user=user_data['user']) # Renamed to avoid conflict later
                
                # Ensure SessionUser entries exist for all provided user_ids
                # Convert session_user_ids to int if they are strings
                processed_user_ids = []
                for user_id_str in session_user_ids:
                    try:
                        processed_user_ids.append(int(user_id_str))
                    except ValueError:
                        logger.error(f"Invalid user_id format in session_user_ids: {user_id_str}")
                        return Response({"error": f"Invalid user_id format: {user_id_str}"}, status=status.HTTP_400_BAD_REQUEST)

                for user_id_val in processed_user_ids:
                    su, created = SessionUser.objects.get_or_create(session=session_instance, user_id=user_id_val)
                    if created:
                        logger.info(f"Created SessionUser: user_id={user_id_val}, session_id={session_instance.id}")
                    else:
                        logger.info(f"SessionUser already exists: user_id={user_id_val}, session_id={session_instance.id}")
                
                # Assign speaker tags based on the order of session_user_ids, fetched by created_at
                # This order matches the one used in transcribe_with_speaker_diarization
                actual_session_users = SessionUser.objects.filter(session=session_instance, user_id__in=processed_user_ids).order_by('created_at')
                for idx, su_instance in enumerate(actual_session_users):
                    tag = f"Speaker_{idx + 1}"
                    if su_instance.speaker_tag != tag: # Only update if different or not set
                        su_instance.speaker_tag = tag
                        su_instance.save()
                        logger.info(f"Assigned speaker_tag '{tag}' to user {su_instance.user.username} in session {session_instance.id}")

            except Session.DoesNotExist:
                logger.error(f"Session not found: ID {session_id}")
                return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
            except Exception as e:
                logger.error(f"Session user creation failed: {str(e)}")
                return Response({"error": f"Session user creation failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            transcription = transcribe_with_speaker_diarization(file_url, MODEL_PATH, SPEAKER_MODEL_PATH, session_id)
            transcription_text = " ".join([segment["text"] for segment in transcription])
            logger.info(f"Transcription segments: {transcription}")
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return Response({"error": f"Transcription failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        audio_instance = AudioFile.objects.create(
            file_path=file_url,
            transcription=transcription,
            status="processed",
            duration=len(transcription_text.split()),
            sop=None,
            user=user_data['user']  # Set the user
        )
        logger.info(f"Created AudioFile instance: {audio_instance.id}")

        # Log audit
        AuditLog.objects.create(
            action='audio_upload',
            user=request.validated_user,
            # user = user_data['user'],
            session_id=session_id,
            object_id=audio_instance.id,
            object_type='AudioFile',
            details={'file_name': audio_file.name}
        )

        response_data = {
            "transcription": transcription,
            "status": "processed",
            "audio_file": AudioFileSerializer(audio_instance).data
        }

        if sop_id:
            try:
                sop = SOP.objects.get(id=sop_id)
                sop_matches = match_sop_steps(transcription_text, sop)
                response_data["sop_matches"] = sop_matches
                audio_instance.sop = sop
                audio_instance.save()
                logger.info(f"SOP matching completed for SOP ID {sop_id}")
            except SOP.DoesNotExist:
                logger.error(f"SOP not found: ID {sop_id}")
                response_data["sop_error"] = "SOP not found"
            except Exception as e:
                logger.error(f"SOP matching failed: {str(e)}")
                response_data["sop_error"] = f"SOP matching failed: {str(e)}"

        if session_id:
            try:
                session = Session.objects.get(id=session_id, user=request.validated_user)
                session.audio_files.add(audio_instance)
                logger.info(f"Audio file {audio_instance.id} added to session {session_id}")
            except Session.DoesNotExist:
                logger.error(f"Session not found or unauthorized: ID {session_id}")
                response_data["session_error"] = "Session not found or unauthorized"
            except Exception as e:
                logger.error(f"Session linking failed: {str(e)}")
                response_data["session_error"] = f"Session linking failed: {str(e)}"

        logger.info("Audio processing completed successfully")
        return Response(response_data, status=status.HTTP_200_OK)


class FeedbackView(APIView):
    permission_classes = [RoleBasedPermission]

    def post(self, request, token, format=None):
        # Validate token
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        # Set validated user for permission class (if needed)
        request.validated_user = user_data['user']

        serializer = FeedbackSerializer(data=request.data)
        if serializer.is_valid():
            feedback = serializer.save()
            # Log audit
            AuditLog.objects.create(
                action='feedback_submit',
                user=user_data['user'],
                object_id=feedback.id,
                object_type='Feedback',
                details={'audio_file_id': feedback.audio_file.id}
            )
            return Response({"message": "Feedback submitted successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class GetAudioRecordsView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token, format=None):
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)
        user = user_data['user']
        try:
            # audio_records = AudioFile.objects.all().order_by("-id")
            if user_data['user'].role == 'admin':
                audio_records = AudioFile.objects.all()
            elif user.role == 'operator':
                audio_records = AudioFile.objects.filter(
                    models.Q(user=user) | models.Q(sessions__user=user)
                ).distinct()
            else:  # reviewer
                audio_records = AudioFile.objects.filter(
                    sessions__session_users__user=user
                ).distinct()
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
        # Validate token
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        # Set validated user for permission class (if needed)
        request.validated_user = user_data['user']

        file_path = request.data.get("file_path")
        new_keywords = request.data.get("new_keywords", "")

        if not file_path or not new_keywords:
            return Response({"error": "File path and keywords are required"}, status=status.HTTP_400_BAD_REQUEST)

        # Process transcription
        transcription = process_audio_pipeline(file_path, MODEL_PATH)
        
        keywords_list = [kw.strip() for kw in new_keywords.split(",")]
        detected_keywords = detect_keywords(transcription, keywords_list)

        AudioFile.objects.filter(file_path=file_path).update(
            transcription=transcription,
            keywords_detected=detected_keywords,
            status="reanalyzed",
            duration=len(transcription.split()),
        )
        return Response(
            {
                "file_path": file_path,
                "transcription": transcription,
                "detected_keywords": detected_keywords,
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
        # Validate token
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            sop = serializer.save(created_by=user_data['user'])
            # Log audit
            AuditLog.objects.create(
                action='sop_create',
                user=user_data['user'],
                object_id=sop.id,
                object_type='SOP',
                details={'name': sop.name, 'version': sop.version}
            )
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SOPListView(APIView):
    permission_classes = [RoleBasedPermission]
    def get(self, request, token):
        # Validate token
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        if user_data['user'].role == 'admin':
                sops = SOP.objects.all()
        else:
            sops = SOP.objects.filter(created_by=user_data['user'])
        serializer = SOPSerializer(sops, many=True)
        return Response({'sops': serializer.data}, status=status.HTTP_200_OK)
    
class SessionCreateView(CreateAPIView):
    serializer_class = SessionSerializer
    permission_classes = [RoleBasedPermission]

    def post(self, request, token, format=None):
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.serializer_class(data=request.data)
        print(f"Request data: {request.data}")
        print(f"User data: {user_data}")
        if serializer.is_valid():
            serializer.save(user=user_data['user'])
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SessionListView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token, format=None):
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        try:
            if user_data['user'].role == 'admin':
                sessions = Session.objects.all()
            else:
                sessions = Session.objects.filter(user=user_data['user'])
            sessions = sessions.order_by("-created_at")
            serializer = SessionSerializer(sessions, many=True)
            return Response({'sessions': serializer.data}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching sessions: {str(e)}")
            return Response({"error": f"Error fetching sessions: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class SessionReviewView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, session_id, token, format=None):
        logger.info(f"Fetching review data for session ID: {session_id}, user: {request.validated_user.username}")
        try:
            session = Session.objects.get(id=session_id)
            serializer = SessionSerializer(session)
            logger.info(f"Retrieved review data for session: {session.name}")
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Session.DoesNotExist:
            logger.error(f"Session not found: ID {session_id}")
            return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error fetching session review: {str(e)}")
            return Response({"error": f"Error fetching session review: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request, session_id, token, format=None):
        logger.info(f"Submitting review for session ID: {session_id}, user: {request.validated_user.username}")
        try:
            session = Session.objects.get(id=session_id)
            data = request.data.copy()
            data['reviewer_id'] = request.validated_user.id
            data['session'] = session.id
            serializer = FeedbackReviewSerializer(data=data)
            if serializer.is_valid():
                feedback_review = serializer.save() # Assign to variable
                # Log audit for FeedbackReview submission
                AuditLog.objects.create(
                    action='review_submit',
                    user=request.validated_user,
                    session=session,
                    object_id=feedback_review.id,
                    object_type='FeedbackReview',
                    details={'reviewer_comments_length': len(feedback_review.comments if feedback_review.comments else "")}
                )
                logger.info("Feedback review submitted successfully")
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            logger.error(f"Feedback review submission failed: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Session.DoesNotExist:
            logger.error(f"Session not found: ID {session_id}")
            return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error submitting feedback review: {str(e)}")
            return Response({"error": f"Error submitting feedback review: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
        logger.info(f"Fetching user settings for user: {request.validated_user.username}")
        try:
            settings, created = UserSettings.objects.get_or_create(user=request.validated_user)
            serializer = UserSettingsSerializer(settings)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching user settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def patch(self, request, token, format=None):
        logger.info(f"Updating user settings for user: {request.validated_user.username}")
        try:
            settings, created = UserSettings.objects.get_or_create(user=request.validated_user)
            serializer = UserSettingsSerializer(settings, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                # Update UserProfile theme if provided
                if 'theme' in request.data:
                    request.validated_user.theme = request.data['theme']
                    request.validated_user.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error updating user settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SystemSettingsView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token, format=None):
        logger.info(f"Fetching system settings for user: {request.validated_user.username}")
        try:
            settings, created = SystemSettings.objects.get_or_create(id=1)
            serializer = SystemSettingsSerializer(settings)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching system settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def patch(self, request, token, format=None):
        logger.info(f"Updating system settings for user: {request.validated_user.username}")
        try:
            settings, created = SystemSettings.objects.get_or_create(id=1)
            serializer = SystemSettingsSerializer(settings, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error updating system settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AuditLogView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token, format=None):
        logger.info(f"Fetching audit logs for user: {request.validated_user.username}")
        try:
            logs = AuditLog.objects.all().order_by('-timestamp')
            serializer = AuditLogSerializer(logs, many=True)
            return Response({'audit_logs': serializer.data}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching audit logs: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SessionStatusUpdateView(APIView):
    permission_classes = [RoleBasedPermission]

    def patch(self, request, session_id, token, format=None):
        # request.validated_user is set by RoleBasedPermission
        user = request.validated_user
        
        session_obj = get_object_or_404(Session, id=session_id) # Renamed to avoid conflict
        requested_status = request.data.get('status')

        if not requested_status or requested_status not in [choice[0] for choice in Session.STATUS_CHOICES]:
            return Response({'error': f"Invalid status value. Must be one of: {[choice[0] for choice in Session.STATUS_CHOICES]}"}, status=status.HTTP_400_BAD_REQUEST)

        if user.role == 'operator' and session_obj.user != user:
            logger.warning(f"Permission denied: Operator {user.username} trying to update session {session_obj.id} owned by {session_obj.user.username}")
            return Response({'error': 'Permission denied. Operators can only update their own sessions.'}, status=status.HTTP_403_FORBIDDEN)

        old_status = session_obj.status # Capture status before update

        # If admin, or operator owns the session, proceed with update
        session_obj.status = requested_status
        session_obj.save()
        
        # Log audit
        AuditLog.objects.create(
            action='session_status_update',
            user=user,
            session=session_obj, # Use the retrieved session object
            object_id=session_obj.id,
            object_type='Session',
            details={'previous_status': old_status, 'new_status': requested_status}
        )
        
        logger.info(f"Session {session_obj.id} status updated to {requested_status} by user {user.username}")
        serializer = SessionSerializer(session_obj) # Use the retrieved session object
        return Response(serializer.data, status=status.HTTP_200_OK)
