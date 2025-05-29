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
        
        current_user = user_data['user']
        # Set validated user on request IF RoleBasedPermission is expected to use it,
        # but rely on current_user from token_verification for this method's logic.
        request.validated_user = current_user 

        serializer = FeedbackSerializer(data=request.data)
        if serializer.is_valid():
            # Save the feedback instance, providing the validated user for created_by
            feedback_obj = serializer.save(created_by=current_user)
            
            # Log audit
            AuditLog.objects.create(
                action='feedback_submit',
                user=current_user,
                object_id=feedback_obj.id,
                object_type='Feedback',
                details={'audio_file_id': feedback_obj.audio_file.id} 
            )
            logger.info(f"Feedback {feedback_obj.id} submitted by user {current_user.username}")
            # Return the serialized feedback object
            return Response(FeedbackSerializer(feedback_obj).data, status=status.HTTP_201_CREATED)
        logger.error(f"Error submitting feedback: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class FeedbackListView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token):
        # request.validated_user is set by RoleBasedPermission
        user = request.validated_user
        queryset = Feedback.objects.all().order_by('-created_at')

        audio_file_id = request.query_params.get('audio_file_id')
        if audio_file_id:
            queryset = queryset.filter(audio_file_id=audio_file_id)

        # Permission filtering
        if user.role == 'admin':
            # Admin can see all feedback (already filtered by audio_file_id if provided)
            pass
        elif user.role == 'operator':
            # Operators can see:
            # 1. Feedback they submitted (created_by=user)
            # 2. Feedback on audio files they uploaded (audio_file__user=user)
            # 3. Feedback on audio files in sessions they are part of/own (audio_file__sessions__user=user)
            queryset = queryset.filter(
                models.Q(created_by=user) | 
                models.Q(audio_file__user=user) |
                models.Q(audio_file__sessions__user=user)
            ).distinct()
        elif user.role == 'reviewer':
            # Reviewers can see feedback on audio files in sessions they are assigned to review
            # (audio_file__sessions__session_users__user=user where SessionUser links to the reviewer)
            queryset = queryset.filter(audio_file__sessions__session_users__user=user).distinct()
        else:
            logger.warning(f"User {user.username} with unhandled role '{user.role}' attempted to list feedback.")
            return Response({"error": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)
            
        serializer = FeedbackSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class FeedbackDetailView(APIView):
    permission_classes = [RoleBasedPermission]

    def get_object(self, feedback_id, user):
        try:
            feedback_obj = Feedback.objects.select_related('audio_file__user', 'created_by').get(pk=feedback_id)
        except Feedback.DoesNotExist:
            logger.info(f"Feedback with id {feedback_id} not found.")
            return None

        # Permission check for individual object read access
        if user.role == 'admin':
            return feedback_obj
        
        can_access = False
        if user.role == 'operator':
            # Operator can access:
            # 1. Feedback they submitted
            # 2. Feedback on audio files they uploaded
            # 3. Feedback on audio files in sessions they own/are part of
            if feedback_obj.created_by == user or \
               (feedback_obj.audio_file and feedback_obj.audio_file.user == user) or \
               (feedback_obj.audio_file and feedback_obj.audio_file.sessions.filter(user=user).exists()):
                can_access = True
        elif user.role == 'reviewer':
            # Reviewer can access feedback if it's on an audio file in a session they are reviewing
            if feedback_obj.audio_file and feedback_obj.audio_file.sessions.filter(session_users__user=user).exists():
                can_access = True
        
        if not can_access:
            logger.warning(f"User {user.username} (role: {user.role}) permission denied for Feedback {feedback_id}")
            return None
            
        return feedback_obj

    def get(self, request, feedback_id, token):
        # request.validated_user is set by RoleBasedPermission
        feedback_obj = self.get_object(feedback_id, request.validated_user)
        if feedback_obj is None:
            return Response({"error": "Feedback not found or permission denied."}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = FeedbackSerializer(feedback_obj)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, feedback_id, token): # Handles full updates
        return self._update_handler(request, feedback_id, token, partial=False)

    def patch(self, request, feedback_id, token): # Handles partial updates
        return self._update_handler(request, feedback_id, token, partial=True)

    def _update_handler(self, request, feedback_id, token, partial):
        user = request.validated_user # Set by RoleBasedPermission
        feedback_obj_for_check = self.get_object(feedback_id, user) 

        if feedback_obj_for_check is None: # Check if user had initial read access
            return Response({"error": "Feedback not found or permission denied for access."}, status=status.HTTP_404_NOT_FOUND)

        # Specific permission check for update/patch: admin or owner (created_by)
        if not (user.role == 'admin' or feedback_obj_for_check.created_by == user):
            logger.warning(f"Permission denied for user {user.username} to update Feedback {feedback_id}")
            return Response({"error": "You do not have permission to update this feedback."}, status=status.HTTP_403_FORBIDDEN)

        # Ensure audio_file_id is not part of the update data.
        # The serializer should handle this via read_only_fields / write_only fields,
        # but an explicit check here is safer.
        if 'audio_file_id' in request.data or 'audio_file' in request.data:
             logger.warning(f"Attempt to modify 'audio_file' or 'audio_file_id' during Feedback update by {user.username} denied.")
             return Response({"error": "Cannot change the associated audio file of a feedback record."}, status=status.HTTP_400_BAD_REQUEST)

        # Fetch the actual object again for update, without re-running all permission checks if already passed
        try:
            feedback_to_update = Feedback.objects.get(pk=feedback_id)
        except Feedback.DoesNotExist:
             return Response({"error": "Feedback not found."}, status=status.HTTP_404_NOT_FOUND) # Should be caught by get_object earlier

        serializer = FeedbackSerializer(feedback_to_update, data=request.data, partial=partial)
        if serializer.is_valid():
            updated_feedback = serializer.save() # created_by is not changed here due to read_only in serializer
            AuditLog.objects.create(
                action='feedback_update',
                user=user,
                object_id=updated_feedback.id,
                object_type='Feedback',
                details={'updated_fields': list(request.data.keys())}
            )
            logger.info(f"Feedback {updated_feedback.id} updated by user {user.username}")
            return Response(FeedbackSerializer(updated_feedback).data, status=status.HTTP_200_OK)
        logger.error(f"Error updating feedback {feedback_id}: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, feedback_id, token):
        user = request.validated_user # Set by RoleBasedPermission
        feedback_obj_for_check = self.get_object(feedback_id, user)

        if feedback_obj_for_check is None: # Check if user had initial read access
            return Response({"error": "Feedback not found or permission denied for access."}, status=status.HTTP_404_NOT_FOUND)

        # Specific permission check for delete: admin or owner (created_by)
        if not (user.role == 'admin' or feedback_obj_for_check.created_by == user):
            logger.warning(f"Permission denied for user {user.username} to delete Feedback {feedback_id}")
            return Response({"error": "You do not have permission to delete this feedback."}, status=status.HTTP_403_FORBIDDEN)
        
        # Fetch the actual object again for delete
        try:
            feedback_to_delete = Feedback.objects.get(pk=feedback_id)
        except Feedback.DoesNotExist:
            return Response({"error": "Feedback not found."}, status=status.HTTP_404_NOT_FOUND)

        audio_file_id_for_log = feedback_to_delete.audio_file.id if feedback_to_delete.audio_file else None
        feedback_to_delete.delete()

        AuditLog.objects.create(
            action='feedback_delete',
            user=user,
            object_id=feedback_id, 
            object_type='Feedback',
            details={'audio_file_id': audio_file_id_for_log}
        )
        logger.info(f"Feedback {feedback_id} deleted by user {user.username}")
        return Response(status=status.HTTP_204_NO_CONTENT)

class AudioFileDetailView(APIView):
    permission_classes = [RoleBasedPermission]

    def get_object(self, audio_id):
        try:
            return AudioFile.objects.get(pk=audio_id)
        except AudioFile.DoesNotExist:
            return None

    def get(self, request, audio_id, token):
        # request.validated_user is set by RoleBasedPermission
        audio_file = self.get_object(audio_id)
        if audio_file is None:
            return Response({"error": "AudioFile not found"}, status=status.HTTP_404_NOT_FOUND)

        # Permission check for GET
        user = request.validated_user
        is_owner = audio_file.user == user
        is_in_session = audio_file.sessions.filter(session_users__user=user).exists()
        
        if not (user.role == 'admin' or is_owner or is_in_session):
            logger.warning(f"Permission denied for user {user.username} to access AudioFile {audio_id}")
            return Response({"error": "You do not have permission to access this audio file."}, status=status.HTTP_403_FORBIDDEN)

        serializer = AudioFileSerializer(audio_file)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, audio_id, token):
        # request.validated_user is set by RoleBasedPermission
        audio_file = self.get_object(audio_id)
        if audio_file is None:
            return Response({"error": "AudioFile not found"}, status=status.HTTP_404_NOT_FOUND)

        user = request.validated_user
        is_owner = audio_file.user == user

        # Permission check for DELETE (admin or owner)
        if not (user.role == 'admin' or is_owner):
            logger.warning(f"Permission denied for user {user.username} to delete AudioFile {audio_id}")
            return Response({"error": "You do not have permission to delete this audio file."}, status=status.HTTP_403_FORBIDDEN)

        file_path = audio_file.file_path
        audio_file_name_for_log = os.path.basename(file_path) # For logging

        # Attempt to delete from S3
        if file_path.startswith(f"https://{S3_BUCKET_NAME}.s3"):
            try:
                # Example: https://my-bucket.s3.amazonaws.com/path/to/file.wav
                # Key should be "path/to/file.wav"
                object_key = file_path.split(f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/")[1]
                s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=object_key)
                logger.info(f"Successfully deleted {object_key} from S3 bucket {S3_BUCKET_NAME}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path} from S3: {str(e)}")
                # Optionally, decide if this error should prevent DB deletion.
                # For now, we log and proceed to delete the DB record.
        else:
            logger.warning(f"File path {file_path} does not seem to be a valid S3 URL for bucket {S3_BUCKET_NAME}. Skipping S3 deletion.")


        audio_file.delete()
        
        # Log audit
        AuditLog.objects.create(
            action='audiofile_delete',
            user=user,
            object_id=audio_id, # audio_file.id is no longer available
            object_type='AudioFile',
            details={'file_name': audio_file_name_for_log, 'file_path': file_path}
        )
        logger.info(f"AudioFile {audio_id} deleted by user {user.username}")
        return Response(status=status.HTTP_204_NO_CONTENT)


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

        # request.validated_user is set by RoleBasedPermission
        if request.validated_user.role == 'admin':
            sops = SOP.objects.all()
        elif request.validated_user.role == 'operator':
            sops = SOP.objects.filter(created_by=request.validated_user)
        elif request.validated_user.role == 'reviewer':
            # Reviewers can see all SOPs for now, or define specific logic if needed
            sops = SOP.objects.all()
        else:
            return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)

        serializer = SOPSerializer(sops, many=True)
        return Response({'sops': serializer.data}, status=status.HTTP_200_OK)

class SOPDetailView(APIView):
    permission_classes = [RoleBasedPermission]

    def get_object(self, sop_id):
        try:
            return SOP.objects.get(pk=sop_id)
        except SOP.DoesNotExist:
            return None

    def get(self, request, sop_id, token):
        # Token validation is handled by RoleBasedPermission
        # request.validated_user is set by RoleBasedPermission
        sop = self.get_object(sop_id)
        if sop is None:
            return Response({"error": "SOP not found"}, status=status.HTTP_404_NOT_FOUND)
        
        # Permission check for GET (admin, operator, reviewer)
        # This is largely handled by RoleBasedPermission, but specific object access can be checked here if needed
        # For now, if RoleBasedPermission allows the view, GET is allowed.

        serializer = SOPSerializer(sop)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, sop_id, token):
        # request.validated_user is set by RoleBasedPermission
        sop = self.get_object(sop_id)
        if sop is None:
            return Response({"error": "SOP not found"}, status=status.HTTP_404_NOT_FOUND)

        # Permission check for PUT (admin or owner)
        if not (request.validated_user.role == 'admin' or sop.created_by == request.validated_user):
            logger.warning(f"Permission denied for user {request.validated_user.username} to update SOP {sop_id}")
            return Response({"error": "You do not have permission to perform this action."}, status=status.HTTP_403_FORBIDDEN)

        serializer = SOPSerializer(sop, data=request.data, partial=False) # Use partial=False for PUT
        if serializer.is_valid():
            updated_sop = serializer.save()
            # Log audit
            AuditLog.objects.create(
                action='sop_update',
                user=request.validated_user,
                object_id=updated_sop.id,
                object_type='SOP',
                details={'name': updated_sop.name, 'version': updated_sop.version, 'updated_fields': list(request.data.keys())}
            )
            logger.info(f"SOP {sop_id} updated by user {request.validated_user.username}")
            return Response(serializer.data, status=status.HTTP_200_OK)
        logger.error(f"Error updating SOP {sop_id}: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, sop_id, token):
        # request.validated_user is set by RoleBasedPermission
        sop = self.get_object(sop_id)
        if sop is None:
            return Response({"error": "SOP not found"}, status=status.HTTP_404_NOT_FOUND)

        # Permission check for DELETE (admin or owner)
        if not (request.validated_user.role == 'admin' or sop.created_by == request.validated_user):
            logger.warning(f"Permission denied for user {request.validated_user.username} to delete SOP {sop_id}")
            return Response({"error": "You do not have permission to perform this action."}, status=status.HTTP_403_FORBIDDEN)

        sop_name = sop.name # For logging before deletion
        sop.delete()
        # Log audit
        AuditLog.objects.create(
            action='sop_delete', # Ensure this action type exists in AuditLog.ACTION_CHOICES
            user=request.validated_user,
            object_id=sop_id, # sop.id is no longer available
            object_type='SOP',
            details={'name': sop_name}
        )
        logger.info(f"SOP {sop_id} deleted by user {request.validated_user.username}")
        return Response(status=status.HTTP_204_NO_CONTENT)
    
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

class FeedbackReviewListView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token):
        user = request.validated_user # Set by RoleBasedPermission
        queryset = FeedbackReview.objects.select_related('reviewer', 'session__user').order_by('-created_at')

        session_id = request.query_params.get('session_id')
        if session_id:
            queryset = queryset.filter(session_id=session_id)

        reviewer_id = request.query_params.get('reviewer_id')
        if reviewer_id:
            queryset = queryset.filter(reviewer_id=reviewer_id)
        
        # Permission filtering
        if user.role == 'admin':
            # Admin can see all reviews (already filtered if params provided)
            pass
        elif user.role == 'operator':
            # Operator can list reviews for their sessions (sessions owned by them)
            queryset = queryset.filter(session__user=user)
        elif user.role == 'reviewer':
            # Reviewer can list reviews they submitted OR reviews for sessions they are a participant in
            queryset = queryset.filter(
                models.Q(reviewer=user) | 
                models.Q(session__session_users__user=user) 
            ).distinct()
        else:
            logger.warning(f"User {user.username} with unhandled role '{user.role}' attempted to list feedback reviews.")
            return Response({"error": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)
            
        serializer = FeedbackReviewSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class FeedbackReviewDetailView(APIView):
    permission_classes = [RoleBasedPermission]

    def get_object_and_check_permission(self, review_id, user, check_write_permission=False):
        try:
            # Use select_related to fetch related objects in a single query
            review = FeedbackReview.objects.select_related('reviewer', 'session__user').get(pk=review_id)
        except FeedbackReview.DoesNotExist:
            logger.info(f"FeedbackReview with id {review_id} not found.")
            return None

        # Admin can do anything
        if user.role == 'admin':
            return review

        is_owner_of_review = (review.reviewer == user)
        # Check if the session being reviewed belongs to the current user (if operator)
        is_owner_of_session_under_review = (review.session.user == user) 

        # For write operations (PUT, PATCH, DELETE) - only Reviewer who owns the review or Admin
        if check_write_permission:
            if user.role == 'reviewer' and is_owner_of_review:
                return review
            # Admins already handled. Operators cannot edit/delete reviews directly via this view.
            logger.warning(f"Write permission denied for user {user.username} (role: {user.role}) on FeedbackReview {review_id}.")
            return None
        
        # For read operations (GET)
        if user.role == 'reviewer':
            # Reviewer can see their own reviews or reviews for sessions they are a participant in
            is_participant_in_session_under_review = SessionUser.objects.filter(session=review.session, user=user).exists()
            if is_owner_of_review or is_participant_in_session_under_review:
                return review
        elif user.role == 'operator':
            # Operator can see reviews on their sessions
            if is_owner_of_session_under_review:
                return review
        
        logger.warning(f"Read permission denied for user {user.username} (role: {user.role}) on FeedbackReview {review_id}.")
        return None

    def get(self, request, review_id, token):
        user = request.validated_user # from RoleBasedPermission
        review = self.get_object_and_check_permission(review_id, user, check_write_permission=False)
        
        if review is None:
            return Response({"error": "FeedbackReview not found or permission denied."}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = FeedbackReviewSerializer(review)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, review_id, token): # Full update
        return self._update_handler(request, review_id, token, partial=False)

    def patch(self, request, review_id, token): # Partial update
        return self._update_handler(request, review_id, token, partial=True)

    def _update_handler(self, request, review_id, token, partial):
        user = request.validated_user
        # Use the permission checking method to ensure user has write access
        review_for_permission_check = self.get_object_and_check_permission(review_id, user, check_write_permission=True)

        if review_for_permission_check is None:
            # Message already logged by get_object_and_check_permission if it was a specific permission denial
            return Response({"error": "FeedbackReview not found or permission denied for update."}, status=status.HTTP_403_FORBIDDEN)

        # Fetch the actual object again to pass to serializer. This ensures we are working with a fresh instance.
        try:
             review_to_update = FeedbackReview.objects.get(pk=review_id)
        except FeedbackReview.DoesNotExist: 
            # This case should ideally be caught by get_object_and_check_permission, but as a safeguard:
            return Response({"error": "FeedbackReview not found."}, status=status.HTTP_404_NOT_FOUND)

        # The serializer's update method prevents changing reviewer and session.
        serializer = FeedbackReviewSerializer(review_to_update, data=request.data, partial=partial)
        if serializer.is_valid():
            updated_review = serializer.save()
            AuditLog.objects.create(
                action='feedbackreview_update',
                user=user,
                session=updated_review.session, 
                object_id=updated_review.id,
                object_type='FeedbackReview',
                details={'updated_fields': list(request.data.keys())}
            )
            logger.info(f"FeedbackReview {updated_review.id} updated by user {user.username}")
            return Response(FeedbackReviewSerializer(updated_review).data, status=status.HTTP_200_OK)
        logger.error(f"Error updating FeedbackReview {review_id}: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, review_id, token):
        user = request.validated_user
        review_for_permission_check = self.get_object_and_check_permission(review_id, user, check_write_permission=True)

        if review_for_permission_check is None:
            return Response({"error": "FeedbackReview not found or permission denied for deletion."}, status=status.HTTP_403_FORBIDDEN)
        
        # Fetch the actual object again for deletion
        try:
            review_to_delete = FeedbackReview.objects.get(pk=review_id)
        except FeedbackReview.DoesNotExist: 
            return Response({"error": "FeedbackReview not found."}, status=status.HTTP_404_NOT_FOUND)

        session_id_for_log = review_to_delete.session.id
        comments_for_log = review_to_delete.comments # Capture before delete
        
        review_to_delete.delete()

        AuditLog.objects.create(
            action='feedbackreview_delete',
            user=user,
            session_id=session_id_for_log, 
            object_id=review_id, 
            object_type='FeedbackReview',
            details={'comments_length': len(comments_for_log if comments_for_log else "")}
        )
        logger.info(f"FeedbackReview {review_id} deleted by user {user.username}")
        return Response(status=status.HTTP_204_NO_CONTENT)

# def transcribe_with_speaker_diarization(audio_url: str, model_path: str, speaker_model_path: str):
#     """
#     Transcribes audio with Vosk and includes speaker diarization.
#
#     Args:
#         audio_url (str): URL of the audio file.
#         model_path (str): Path to the Vosk ASR model.
#         speaker_model_path (str): Path to the Vosk speaker diarization model.
#
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
#
#     try:
#         model = vosk.Model(model_path)
#         spk_model = vosk.SpkModel(speaker_model_path)  # Load Speaker Model
#
#         with wave.open(temp_audio_path, "rb") as wf:
#             recognizer = vosk.KaldiRecognizer(model, wf.getframerate())
#             recognizer.SetWords(True)
#             recognizer.SetSpkModel(spk_model)  # Enable speaker diarization
#
#             transcription = []
#
#             while True:
#                 data = wf.readframes(4000)
#                 if len(data) == 0:
#                     break
#
#                 if recognizer.AcceptWaveform(data):
#                     result = json.loads(recognizer.Result())
#
#                     # Extract speaker ID
#                     speaker_id = None
#                     if "spk" in result:
#                         if isinstance(result["spk"], list) and result["spk"]:  # Ensure it's a non-empty list
#                             speaker_id = result["spk"][0]  # Take the first speaker ID
#                         elif isinstance(result["spk"], (int, float)):  # Handle single value
#                             speaker_id = result["spk"]
#
#                     # Ensure valid speaker ID
#                     speaker = f"Speaker_{int(speaker_id) + 1}" if speaker_id is not None else "Unknown"
#
#                     # Append transcription
#                     if "text" in result and result["text"]:
#                         transcription.append({"speaker": speaker, "text": result["text"]})
#
#             # Process the final result
#             final_result = json.loads(recognizer.FinalResult())
#             if "text" in final_result and final_result["text"]:
#                 speaker_id = None
#                 if "spk" in final_result:
#                     if isinstance(final_result["spk"], list) and final_result["spk"]:
#                         speaker_id = final_result["spk"][0]
#                     elif isinstance(final_result["spk"], (int, float)):
#                         speaker_id = final_result["spk"]
#
#                 speaker = f"Speaker_{int(speaker_id) + 1}" if speaker_id is not None else "Unknown"
#                 transcription.append({"speaker": speaker, "text": final_result["text"]})
#
#     finally:
#         os.remove(temp_audio_path)  # Delete temp file after processing
#
#     return transcription

class AdminUserListView(APIView):
    permission_classes = [RoleBasedPermission] # Ensures admin-only access

    def get(self, request, token):
        # request.validated_user is set by RoleBasedPermission, confirming admin role
        if request.validated_user.role != 'admin':
             # This check is technically redundant if RoleBasedPermission is correctly configured
             # to only allow admins for views not specified for other roles, but good for defense in depth.
            return Response({"error": "Forbidden. Admin access required."}, status=status.HTTP_403_FORBIDDEN)

        users = UserProfile.objects.all().order_by('id')
        serializer = AdminUserProfileSerializer(users, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class AdminUserDetailView(APIView):
    permission_classes = [RoleBasedPermission] # Ensures admin-only access

    def get_object(self, user_id):
        try:
            return UserProfile.objects.get(pk=user_id)
        except UserProfile.DoesNotExist:
            return None

    def get(self, request, user_id, token):
        if request.validated_user.role != 'admin':
            return Response({"error": "Forbidden. Admin access required."}, status=status.HTTP_403_FORBIDDEN)
        
        user_profile = self.get_object(user_id)
        if user_profile is None:
            return Response({"error": "UserProfile not found."}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = AdminUserProfileSerializer(user_profile)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, user_id, token): # Full update
        return self._update_handler(request, user_id, token, partial=False)

    def patch(self, request, user_id, token): # Partial update
        return self._update_handler(request, user_id, token, partial=True)

    def _update_handler(self, request, user_id, token, partial):
        if request.validated_user.role != 'admin':
            return Response({"error": "Forbidden. Admin access required."}, status=status.HTTP_403_FORBIDDEN)

        user_profile = self.get_object(user_id)
        if user_profile is None:
            return Response({"error": "UserProfile not found."}, status=status.HTTP_404_NOT_FOUND)

        # Prevent admin from updating their own critical fields like role or is_active via this view
        # to avoid self-lockout. They should use Django admin or other means.
        if user_profile == request.validated_user and ('role' in request.data or 'is_active' in request.data):
            logger.warning(f"Admin user {request.validated_user.username} attempted to change their own role or active status via API.")
            return Response({"error": "Admins cannot change their own role or active status via this API."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Password update is excluded by AdminUserProfileSerializer's update method
        serializer = AdminUserProfileSerializer(user_profile, data=request.data, partial=partial)
        if serializer.is_valid():
            updated_user_profile = serializer.save()
            AuditLog.objects.create(
                action='userprofile_update',
                user=request.validated_user, # The admin performing the action
                object_id=updated_user_profile.id,
                object_type='UserProfile',
                details={
                    'target_user_id': updated_user_profile.id,
                    'target_username': updated_user_profile.username,
                    'updated_fields': list(request.data.keys())
                }
            )
            logger.info(f"UserProfile {updated_user_profile.id} ({updated_user_profile.username}) updated by admin {request.validated_user.username}")
            return Response(AdminUserProfileSerializer(updated_user_profile).data, status=status.HTTP_200_OK)
        logger.error(f"Error updating UserProfile {user_id}: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, user_id, token):
        if request.validated_user.role != 'admin':
            return Response({"error": "Forbidden. Admin access required."}, status=status.HTTP_403_FORBIDDEN)

        user_profile_to_delete = self.get_object(user_id)
        if user_profile_to_delete is None:
            return Response({"error": "UserProfile not found."}, status=status.HTTP_404_NOT_FOUND)

        # Prevent admin from deleting themselves
        if user_profile_to_delete == request.validated_user:
            logger.warning(f"Admin user {request.validated_user.username} attempted to delete themselves via API.")
            return Response({"error": "Admin users cannot delete themselves via the API."}, status=status.HTTP_400_BAD_REQUEST)

        username_for_log = user_profile_to_delete.username
        user_profile_to_delete.delete()

        AuditLog.objects.create(
            action='userprofile_delete',
            user=request.validated_user, # The admin performing the action
            object_id=user_id, 
            object_type='UserProfile',
            details={'deleted_user_id': user_id, 'deleted_username': username_for_log}
        )
        logger.info(f"UserProfile {user_id} ({username_for_log}) deleted by admin {request.validated_user.username}")
        return Response(status=status.HTTP_204_NO_CONTENT)

# def transcribe_with_speaker_diarization(audio_url: str, model_path: str, speaker_model_path: str):
#     """

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

class SessionDetailView(APIView):
    permission_classes = [RoleBasedPermission]

    def get_object_and_check_permission(self, session_id, user, check_ownership_for_write=False):
        try:
            session = Session.objects.select_related('user', 'sop').prefetch_related('audio_files', 'session_users__user').get(pk=session_id)
        except Session.DoesNotExist:
            logger.info(f"Session with id {session_id} not found.")
            return None

        # Admin can do anything
        if user.role == 'admin':
            return session

        is_owner = (session.user == user)
        is_participant = SessionUser.objects.filter(session=session, user=user).exists()
        
        # For write operations (PUT, PATCH, DELETE), only owner (Operator) or Admin can proceed
        if check_ownership_for_write:
            if user.role == 'operator' and is_owner:
                return session
            else: # Admin case already handled, so this denies non-owner operators and all reviewers
                logger.warning(f"Write permission denied for user {user.username} (role: {user.role}) on Session {session_id}. Not owner.")
                return None 
        
        # For read operations (GET)
        if user.role == 'operator':
            if is_owner or is_participant:
                return session
        elif user.role == 'reviewer':
            # Reviewers can retrieve sessions they are a participant in OR are assigned to review via FeedbackReview
            # (Assuming FeedbackReview links a reviewer to a session)
            # For now, let's stick to participant status for reviewers as well, or if they have reviewed it.
            # is_reviewer_for_session = FeedbackReview.objects.filter(session=session, reviewer=user).exists()
            if is_participant: #  or is_reviewer_for_session:
                return session
        
        logger.warning(f"Read permission denied for user {user.username} (role: {user.role}) on Session {session_id}.")
        return None


    def get(self, request, session_id, token):
        user = request.validated_user # Set by RoleBasedPermission
        session = self.get_object_and_check_permission(session_id, user, check_ownership_for_write=False)
        
        if session is None:
            return Response({"error": "Session not found or permission denied."}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = SessionSerializer(session)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, session_id, token):
        return self._update_handler(request, session_id, token, partial=False)

    def patch(self, request, session_id, token):
        return self._update_handler(request, session_id, token, partial=True)

    def _update_handler(self, request, session_id, token, partial):
        user = request.validated_user
        session = self.get_object_and_check_permission(session_id, user, check_ownership_for_write=True)

        if session is None:
            return Response({"error": "Session not found or permission denied for update."}, status=status.HTTP_403_FORBIDDEN) # 403 for permission issue

        # Prevent 'user' (owner) from being changed
        if 'user' in request.data or 'user_id' in request.data:
            logger.warning(f"User {user.username} attempt to change session owner denied for session {session_id}.")
            return Response({"error": "Session owner cannot be changed."}, status=status.HTTP_400_BAD_REQUEST)

        serializer = SessionSerializer(session, data=request.data, partial=partial)
        if serializer.is_valid():
            updated_session = serializer.save()
            AuditLog.objects.create(
                action='session_update',
                user=user,
                session=updated_session,
                object_id=updated_session.id,
                object_type='Session',
                details={'updated_fields': list(request.data.keys())}
            )
            logger.info(f"Session {updated_session.id} updated by user {user.username}")
            return Response(SessionSerializer(updated_session).data, status=status.HTTP_200_OK)
        logger.error(f"Error updating session {session_id}: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, session_id, token):
        user = request.validated_user
        session = self.get_object_and_check_permission(session_id, user, check_ownership_for_write=True)

        if session is None:
            return Response({"error": "Session not found or permission denied for deletion."}, status=status.HTTP_403_FORBIDDEN) # 403 for permission

        session_name_for_log = session.name
        session.delete()

        AuditLog.objects.create(
            action='session_delete',
            user=user,
            # session field in AuditLog might be an issue if it's set to CASCADE on SessionUser etc.
            # For now, we log object_id and type. If session object itself is needed, consider how.
            object_id=session_id, 
            object_type='Session',
            details={'name': session_name_for_log}
        )
        logger.info(f"Session {session_id} named '{session_name_for_log}' deleted by user {user.username}")
        return Response(status=status.HTTP_204_NO_CONTENT)


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
