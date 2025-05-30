import os
import uuid
import boto3
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework import status
from .models import AudioFile, SOP, SOPStep, Session, SessionUser, UserSettings, SystemSettings, AuditLog, Feedback, FeedbackReview, UserProfile
from .serializers import (AudioFileSerializer, FeedbackSerializer, ProcessAudioViewSerializer, 
        SOPSerializer, SessionSerializer,FeedbackReviewSerializer, UserSettingsSerializer, SystemSettingsSerializer, AuditLogSerializer, AdminUserProfileSerializer)
from .utils import *
from peercheck import settings
from fuzzywuzzy import fuzz
from Levenshtein import distance
from .authentication import token_verification
from .permissions import RoleBasedPermission
from rest_framework.pagination import PageNumberPagination 
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes
from .serializers import ( # Ensure all relevant serializers are imported
    AudioFileSerializer, FeedbackSerializer, ProcessAudioViewSerializer, 
    SOPSerializer, SessionSerializer, FeedbackReviewSerializer, 
    UserSettingsSerializer, SystemSettingsSerializer, AuditLogSerializer,
    AdminUserProfileSerializer, ErrorResponseSerializer, LoginSerializer, UserProfileSerializer, # Added ErrorResponseSerializer, LoginSerializer, UserProfileSerializer
)
import logging
from django.db import models
import ast

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
    """
    Handles audio file uploads, processing (transcription, SOP matching), and linking to sessions.
    """
    serializer_class = ProcessAudioViewSerializer
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Upload and Process Audio",
        description="Uploads an audio file, transcribes it, optionally matches it against an SOP, and links it to a session.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=ProcessAudioViewSerializer,
        responses={
            200: OpenApiTypes.OBJECT, # A more specific serializer might be defined for this complex response
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=['Audio Processing']
    )
    def post(self, request, token,format=None):
        """
        Processes an uploaded audio file. This includes:
        - Uploading the file to S3.
        - Transcribing the audio using speaker diarization.
        - Optionally matching transcription against SOP steps if sop_id is provided.
        - Optionally linking the audio file to a session if session_id and session_user_ids are provided.
        - Requires 'operator' role.
        """
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

        # New parameters for enhanced speaker diarization
        min_speaker_duration = float(request.data.get("min_speaker_duration", 2.0))
        speaker_similarity_threshold = float(request.data.get("speaker_similarity_threshold", 0.85))

        # Validate audio file before processing
        if audio_file:
            # Get file extension for format validation
            file_extension = audio_file.name.split('.')[-1].lower() if '.' in audio_file.name else 'unknown'
            supported_formats = get_supported_audio_formats()
            
            if file_extension not in supported_formats and file_extension != 'unknown':
                return Response({
                    "error": f"Unsupported audio format: {file_extension}. "
                             f"Supported formats: {', '.join(supported_formats)}"
                }, status=status.HTTP_400_BAD_REQUEST)


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
                        user_id_int = int(user_id_str)
                        # Verify user exists
                        from django.contrib.auth import get_user_model
                        User = get_user_model()
                        if User.objects.filter(id=user_id_int).exists():
                            processed_user_ids.append(user_id_int)
                        else:
                            logger.warning(f"User ID {user_id_int} not found")
                    except ValueError:
                        logger.error(f"Invalid user_id format: {user_id_str}")
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
            transcription = transcribe_with_speaker_diarization(
                audio_url=file_url, 
                model_path=MODEL_PATH, 
                speaker_model_path=SPEAKER_MODEL_PATH, 
                session_id=session_id,
                min_speaker_duration=min_speaker_duration,
                speaker_similarity_threshold=speaker_similarity_threshold
            )
            transcription_text = " ".join([segment["text"] for segment in transcription])
            # Extract speaker statistics
            unique_speakers = set(segment["speaker"] for segment in transcription)
            speaker_stats = {
                "total_speakers": len(unique_speakers),
                "speakers": list(unique_speakers),
                "total_segments": len(transcription),
                "average_confidence": sum(segment.get("confidence", 0) for segment in transcription) / len(transcription) if transcription else 0
            }
            logger.info(f"Transcription segments: {transcription}")
        except Exception as e:
            logger.error(f"Enhanced transcription failed: {str(e)}")
            # Fallback to basic transcription if enhanced fails
            try:
                logger.info("Attempting fallback to basic transcription")
                transcription_text = process_audio_pipeline(file_url, MODEL_PATH)
                transcription = [{"speaker": "Unknown", "text": transcription_text, "timestamp": 0, "confidence": 0.5}]
                speaker_stats = {"total_speakers": 1, "speakers": ["Unknown"], "total_segments": 1, "average_confidence": 0.5}
                logger.info("Fallback transcription completed")
            except Exception as fallback_error:
                logger.error(f"Both enhanced and fallback transcription failed: {str(fallback_error)}")
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
            user=user_data['user'],
            # user = user_data['user'],
            session_id=session_id,
            object_id=audio_instance.id,
            object_type='AudioFile',
            details={
                    'file_name': audio_file.name,
                    'file_size': audio_file.size,
                    'speaker_stats': speaker_stats,
                    'processing_method': 'enhanced_diarization'
                }
        )

        response_data = {
            "transcription": transcription,
            "transcription_text": transcription_text,  # For backward compatibility
            "speaker_statistics": speaker_stats,
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
    """
    Handles the submission of feedback for an audio file.
    This view is primarily for creating new feedback instances.
    List, Retrieve, Update, Delete operations are handled by FeedbackListView and FeedbackDetailView.
    """
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Submit Feedback",
        description="Submits feedback for a specific audio file. Requires 'operator' role.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=FeedbackSerializer,
        responses={
            201: FeedbackSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
        },
        tags=['Feedback']
    )
    def post(self, request, token, format=None):
        """
        Creates a new feedback entry for an audio file.
        The `created_by` field is automatically set to the authenticated user.
        Requires 'operator' role.
        """
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
    
    @extend_schema(
        summary="List Feedback",
        description="Retrieves a paginated list of feedback records. Filters by user role and allows filtering by `audio_file_id`.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.'),
            OpenApiParameter('audio_file_id', OpenApiTypes.INT, OpenApiParameter.QUERY, description='Optional. Filter feedback by a specific audio file ID.'),
            OpenApiParameter('page', OpenApiTypes.INT, OpenApiParameter.QUERY, description='Page number for pagination.'),
        ],
        responses={
            200: FeedbackSerializer(many=True), # Actual response is paginated
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
        },
        tags=['Feedback']
    )
    def get(self, request, token):
        """
        Lists feedback records with pagination.
        Permissions:
        - Admin: Can see all feedback.
        - Operator: Can see feedback they submitted, or on their audio files, or on audio files in their sessions.
        - Reviewer: Can see feedback on audio files in sessions they are reviewing.
        """
        user = request.validated_user
        queryset = Feedback.objects.all().order_by('-created_at')

        audio_file_id = request.query_params.get('audio_file_id')
        if audio_file_id:
            queryset = queryset.filter(audio_file_id=audio_file_id)

        if user.role == 'admin':
            pass
        elif user.role == 'operator':
            queryset = queryset.filter(
                models.Q(created_by=user) | 
                models.Q(audio_file__user=user) |
                models.Q(audio_file__sessions__user=user)
            ).distinct()
        elif user.role == 'reviewer':
            queryset = queryset.filter(audio_file__sessions__session_users__user=user).distinct()
        else:
            logger.warning(f"User {user.username} with unhandled role '{user.role}' attempted to list feedback.")
            return Response({"error": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)
            
        paginator = PageNumberPagination()
        paginated_queryset = paginator.paginate_queryset(queryset, request, view=self)
        serializer = FeedbackSerializer(paginated_queryset, many=True)
        return paginator.get_paginated_response(serializer.data)

class FeedbackDetailView(APIView):
    """
    Handles Retrieve, Update, and Delete operations for a specific Feedback instance.
    """
    permission_classes = [RoleBasedPermission]

    # get_object method already exists

    @extend_schema(
        summary="Retrieve Feedback Details",
        description="Fetches the details of a specific feedback record by its ID.",
        parameters=[
            OpenApiParameter('feedback_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the feedback to retrieve.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            200: FeedbackSerializer,
            404: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
        },
        tags=['Feedback']
    )
    def get(self, request, feedback_id, token):
        """
        Retrieve a specific feedback record.
        Permissions are checked by the `get_object` helper based on user role and relation to the feedback/audio file.
        """
        feedback_obj = self.get_object(feedback_id, request.validated_user)
        if feedback_obj is None:
            return Response({"error": "Feedback not found or permission denied."}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = FeedbackSerializer(feedback_obj)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @extend_schema(
        summary="Update Feedback",
        description="Updates an existing feedback record. Requires 'admin' role or feedback ownership.",
        parameters=[
            OpenApiParameter('feedback_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the feedback to update.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=FeedbackSerializer, # Note: audio_file_id should not be updatable via this
        responses={
            200: FeedbackSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['Feedback']
    )
    def put(self, request, feedback_id, token):
        """Handles full updates for a feedback record."""
        return self._update_handler(request, feedback_id, token, partial=False)

    @extend_schema(
        summary="Partially Update Feedback",
        description="Partially updates an existing feedback record. Requires 'admin' role or feedback ownership.",
        parameters=[
            OpenApiParameter('feedback_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the feedback to partially update.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=FeedbackSerializer, # Note: audio_file_id should not be updatable
        responses={
            200: FeedbackSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['Feedback']
    )
    def patch(self, request, feedback_id, token):
        """Handles partial updates for a feedback record."""
        return self._update_handler(request, feedback_id, token, partial=True)

    # _update_handler method already exists

    @extend_schema(
        summary="Delete Feedback",
        description="Deletes an existing feedback record. Requires 'admin' role or feedback ownership.",
        parameters=[
            OpenApiParameter('feedback_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the feedback to delete.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            204: OpenApiTypes.NONE,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['Feedback']
    )
    def delete(self, request, feedback_id, token):
        """
        Deletes a feedback record.
        Requires 'admin' role or the user to be the creator of the feedback.
        """
        user = request.validated_user 
        feedback_obj_for_check = self.get_object(feedback_id, user)

        if feedback_obj_for_check is None: 
            return Response({"error": "Feedback not found or permission denied for access."}, status=status.HTTP_404_NOT_FOUND)

        if not (user.role == 'admin' or feedback_obj_for_check.created_by == user):
            logger.warning(f"Permission denied for user {user.username} to delete Feedback {feedback_id}")
            return Response({"error": "You do not have permission to delete this feedback."}, status=status.HTTP_403_FORBIDDEN)
        
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

class FeedbackReviewListView(APIView):
    """
    Retrieves a list of Feedback Reviews.
    Supports pagination and role-based filtering.
    """
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="List Feedback Reviews",
        description="Fetches a paginated list of feedback reviews. Filters by user role and allows filtering by `session_id` or `reviewer_id`.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.'),
            OpenApiParameter('session_id', OpenApiTypes.INT, OpenApiParameter.QUERY, description='Optional. Filter reviews by a specific session ID.'),
            OpenApiParameter('reviewer_id', OpenApiTypes.INT, OpenApiParameter.QUERY, description='Optional. Filter reviews by a specific reviewer ID.'),
            OpenApiParameter('page', OpenApiTypes.INT, OpenApiParameter.QUERY, description='Page number for pagination.'),
        ],
        responses={
            200: FeedbackReviewSerializer(many=True), # Actual response is paginated
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
        },
        tags=['Feedback Review']
    )
    def get(self, request, token):
        """
        Lists feedback reviews with pagination.
        Permissions:
        - Admin: Can see all reviews.
        - Operator: Can list reviews for their sessions.
        - Reviewer: Can list reviews they submitted or reviews for sessions they participate in.
        """
        user = request.validated_user
        queryset = FeedbackReview.objects.select_related('reviewer', 'session__user').order_by('-created_at')

        session_id = request.query_params.get('session_id')
        if session_id:
            queryset = queryset.filter(session_id=session_id)

        reviewer_id = request.query_params.get('reviewer_id')
        if reviewer_id:
            queryset = queryset.filter(reviewer_id=reviewer_id)
        
        if user.role == 'admin':
            pass
        elif user.role == 'operator':
            queryset = queryset.filter(session__user=user)
        elif user.role == 'reviewer':
            queryset = queryset.filter(
                models.Q(reviewer=user) | 
                models.Q(session__session_users__user=user) 
            ).distinct()
        else:
            logger.warning(f"User {user.username} with unhandled role '{user.role}' attempted to list feedback reviews.")
            return Response({"error": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)
            
        paginator = PageNumberPagination()
        paginated_queryset = paginator.paginate_queryset(queryset, request, view=self)
        serializer = FeedbackReviewSerializer(paginated_queryset, many=True)
        return paginator.get_paginated_response(serializer.data)

class FeedbackReviewDetailView(APIView):
    """
    Handles Retrieve, Update, and Delete operations for a specific FeedbackReview instance.
    """
    permission_classes = [RoleBasedPermission]

    # get_object_and_check_permission method already exists

    @extend_schema(
        summary="Retrieve Feedback Review Details",
        description="Fetches the details of a specific feedback review by its ID.",
        parameters=[
            OpenApiParameter('review_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the feedback review to retrieve.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            200: FeedbackReviewSerializer,
            404: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
        },
        tags=['Feedback Review']
    )
    def get(self, request, review_id, token):
        """
        Retrieve a specific feedback review.
        Permissions checked by `get_object_and_check_permission`.
        """
        user = request.validated_user
        review = self.get_object_and_check_permission(review_id, user, check_write_permission=False)
        
        if review is None:
            return Response({"error": "FeedbackReview not found or permission denied."}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = FeedbackReviewSerializer(review)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @extend_schema(
        summary="Update Feedback Review",
        description="Updates an existing feedback review. Requires 'admin' role or review ownership (Reviewer).",
        parameters=[
            OpenApiParameter('review_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the feedback review to update.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=FeedbackReviewSerializer, # Note: reviewer and session are not updatable via serializer
        responses={
            200: FeedbackReviewSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['Feedback Review']
    )
    def put(self, request, review_id, token):
        """Handles full updates for a feedback review."""
        return self._update_handler(request, review_id, token, partial=False)

    @extend_schema(
        summary="Partially Update Feedback Review",
        description="Partially updates an existing feedback review. Requires 'admin' role or review ownership (Reviewer).",
        parameters=[
            OpenApiParameter('review_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the feedback review to partially update.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=FeedbackReviewSerializer, # Note: reviewer and session are not updatable
        responses={
            200: FeedbackReviewSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['Feedback Review']
    )
    def patch(self, request, review_id, token):
        """Handles partial updates for a feedback review."""
        return self._update_handler(request, review_id, token, partial=True)

    # _update_handler method already exists

    @extend_schema(
        summary="Delete Feedback Review",
        description="Deletes an existing feedback review. Requires 'admin' role or review ownership (Reviewer).",
        parameters=[
            OpenApiParameter('review_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the feedback review to delete.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            204: OpenApiTypes.NONE,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['Feedback Review']
    )
    def delete(self, request, review_id, token):
        """
        Deletes a feedback review.
        Requires 'admin' role or the user to be the creator of the review.
        """
        user = request.validated_user
        review_for_permission_check = self.get_object_and_check_permission(review_id, user, check_write_permission=True)

        if review_for_permission_check is None:
            return Response({"error": "FeedbackReview not found or permission denied for deletion."}, status=status.HTTP_403_FORBIDDEN)
        
        try:
            review_to_delete = FeedbackReview.objects.get(pk=review_id)
        except FeedbackReview.DoesNotExist: 
            return Response({"error": "FeedbackReview not found."}, status=status.HTTP_404_NOT_FOUND)

        session_id_for_log = review_to_delete.session.id
        comments_for_log = review_to_delete.comments 
        
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

class AudioFileDetailView(APIView):
    permission_classes = [RoleBasedPermission]

    def get_object(self, audio_id):
        """Helper method to get AudioFile object by ID."""
        try:
            return AudioFile.objects.get(pk=audio_id)
        except AudioFile.DoesNotExist:
            return None

    @extend_schema(
        summary="Retrieve Audio File Details",
        description="Fetches the details of a specific audio file by its ID.",
        parameters=[
            OpenApiParameter('audio_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the audio file to retrieve.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            200: AudioFileSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
        },
        tags=['Audio Records']
    )
    def get(self, request, audio_id, token):
        """
        Retrieve a specific AudioFile by its ID.
        Permissions:
        - Admin: Can access any audio file.
        - Operator: Can access their own audio files or those in their sessions.
        - Reviewer: Can access audio files in sessions they are part of.
        """
        audio_file = self.get_object(audio_id)
        if audio_file is None:
            return Response({"error": "AudioFile not found"}, status=status.HTTP_404_NOT_FOUND)

        user = request.validated_user
        is_owner = audio_file.user == user
        is_in_session = audio_file.sessions.filter(session_users__user=user).exists()
        
        if not (user.role == 'admin' or is_owner or is_in_session):
            logger.warning(f"Permission denied for user {user.username} to access AudioFile {audio_id}")
            return Response({"error": "You do not have permission to access this audio file."}, status=status.HTTP_403_FORBIDDEN)

        serializer = AudioFileSerializer(audio_file)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @extend_schema(
        summary="Delete Audio File",
        description="Deletes an audio file, including its S3 object. Requires 'admin' role or ownership of the audio file.",
        parameters=[
            OpenApiParameter('audio_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the audio file to delete.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            204: OpenApiTypes.NONE,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
        },
        tags=['Audio Records']
    )
    def delete(self, request, audio_id, token):
        """
        Deletes an AudioFile.
        Requires 'admin' role or the user to be the uploader of the audio file.
        Also attempts to delete the corresponding file from S3.
        """
        audio_file = self.get_object(audio_id)
        if audio_file is None:
            return Response({"error": "AudioFile not found"}, status=status.HTTP_404_NOT_FOUND)

        user = request.validated_user
        is_owner = audio_file.user == user

        if not (user.role == 'admin' or is_owner):
            logger.warning(f"Permission denied for user {user.username} to delete AudioFile {audio_id}")
            return Response({"error": "You do not have permission to delete this audio file."}, status=status.HTTP_403_FORBIDDEN)

        file_path = audio_file.file_path
        audio_file_name_for_log = os.path.basename(file_path) 

        if file_path.startswith(f"https://{S3_BUCKET_NAME}.s3"):
            try:
                object_key = file_path.split(f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/")[1]
                s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=object_key)
                logger.info(f"Successfully deleted {object_key} from S3 bucket {S3_BUCKET_NAME}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path} from S3: {str(e)}")
        else:
            logger.warning(f"File path {file_path} does not seem to be a valid S3 URL for bucket {S3_BUCKET_NAME}. Skipping S3 deletion.")

        audio_file.delete()
        
        AuditLog.objects.create(
            action='audiofile_delete',
            user=user,
            object_id=audio_id,
            object_type='AudioFile',
            details={'file_name': audio_file_name_for_log, 'file_path': file_path}
        )
        logger.info(f"AudioFile {audio_id} deleted by user {user.username}")
        return Response(status=status.HTTP_204_NO_CONTENT)


class GetAudioRecordsView(APIView):
    """
    Retrieves a list of audio records based on user role and permissions.
    Supports pagination.
    """
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="List Audio Records",
        description="Fetches a paginated list of audio records. Access is role-dependent: Admins see all, Operators see their own or those in their sessions, Reviewers see those in sessions they are part of.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.'),
            OpenApiParameter('page', OpenApiTypes.INT, OpenApiParameter.QUERY, description='Page number for pagination.'),
        ],
        responses={
            200: AudioFileSerializer(many=True), # Note: actual response is paginated
            401: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=['Audio Records']
    )
    def get(self, request, token, format=None):
        """
        Lists audio records with pagination.
        - Admins: Can see all audio records.
        - Operators: Can see audio records they uploaded or those part of their sessions.
        - Reviewers: Can see audio records part of sessions they are assigned to.
        """
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST) # Should be caught by permission class ideally
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
            
            paginator = PageNumberPagination()
            paginated_queryset = paginator.paginate_queryset(audio_records, request, view=self)
            serializer = AudioFileSerializer(paginated_queryset, many=True)
            response_data = serializer.data

            # Convert 'keywords_detected' from string to list for each item
            for item in response_data:
                keywords = item.get('keywords_detected', None)
                if isinstance(keywords, str):
                    try:
                        item['keywords_detected'] = ast.literal_eval(keywords)
                    except (ValueError, SyntaxError):
                        # If parsing fails, assign an empty list or handle accordingly
                        item['keywords_detected'] = []

            return paginator.get_paginated_response(response_data)
        except Exception as e:
            logger.error(f"Error fetching audio records with pagination: {str(e)}")
            return Response(
                {"error": f"Error fetching audio records: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ReAnalyzeAudioView(APIView):
    """
    Handles re-analysis of an existing audio file, typically with new keywords.
    """
    permission_classes = [RoleBasedPermission]
    # Define a simple serializer for the request body if not complex
    # class ReAnalyzeAudioRequestSerializer(serializers.Serializer):
    #     file_path = serializers.CharField()
    #     new_keywords = serializers.CharField()

    @extend_schema(
        summary="Re-analyze Audio",
        description="Triggers a re-analysis of a previously processed audio file using new keywords. Requires 'operator' role.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=OpenApiTypes.OBJECT, # Or a specific ReAnalyzeAudioRequestSerializer
        # example request: {'file_path': 's3://path/to/file.wav', 'new_keywords': 'safety,procedure'}
        responses={
            200: OpenApiTypes.OBJECT, # Define a specific response serializer if structure is consistent
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
        },
        tags=['Audio Processing']
    )
    def post(self, request, token):
        """
        Re-analyzes an audio file.
        Updates transcription, detected keywords, and status.
        Requires 'operator' role.
        """
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


        try:
            audio_instance = AudioFile.objects.get(file_path=file_path)
        except AudioFile.DoesNotExist:
            return Response({"error": "Audio file not found"}, status=status.HTTP_404_NOT_FOUND)
        
        # New parameters for enhanced speaker diarization
        min_speaker_duration = float(request.data.get("min_speaker_duration", 2.0))
        speaker_similarity_threshold = float(request.data.get("speaker_similarity_threshold", 0.85))
    
        # Initialize variables
        transcription = None
        transcription_text = ""
        speaker_stats = {}
        try:
            # Call your speaker diarization transcription function
            transcription = transcribe_with_speaker_diarization(
                audio_url=file_path,
                model_path=MODEL_PATH,
                speaker_model_path=SPEAKER_MODEL_PATH,
                session_id=None,  # or generate a session id if needed
                min_speaker_duration=min_speaker_duration,
                speaker_similarity_threshold=speaker_similarity_threshold
            )
            # Compose full transcription text
            transcription_text = " ".join([segment["text"] for segment in transcription])

            # Extract speaker statistics
            unique_speakers = set(segment["speaker"] for segment in transcription)
            total_confidence = sum(segment.get("confidence", 0) for segment in transcription)
            average_confidence = total_confidence / len(transcription) if transcription else 0

            speaker_stats = {
                "total_speakers": len(unique_speakers),
                "speakers": list(unique_speakers),
                "total_segments": len(transcription),
                "average_confidence": average_confidence
            }

            logger.info(f"Transcription segments: {transcription}")

        except Exception as e:
            logger.error(f"Enhanced transcription failed: {str(e)}")
            # Fallback to basic transcription
            try:
                logger.info("Attempting fallback to basic transcription")
                transcription_text = process_audio_pipeline(file_path, MODEL_PATH)
                transcription = [{"speaker": "Unknown", "text": transcription_text, "timestamp": 0, "confidence": 0.5}]
                speaker_stats = {
                    "total_speakers": 1,
                    "speakers": ["Unknown"],
                    "total_segments": 1,
                    "average_confidence": 0.5
                }
                logger.info("Fallback transcription completed")
            except Exception as fallback_error:
                logger.error(f"Both enhanced and fallback transcription failed: {str(fallback_error)}")
                return Response({"error": f"Transcription failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        # # Process transcription
        # transcription = process_audio_pipeline(file_path, MODEL_PATH)
        
        # keywords_list = [kw.strip() for kw in new_keywords.split(",")]
        # detected_keywords = detect_keywords(transcription, keywords_list)

        # Save results to database
        try:
            # Assuming transcription is a JSONField in your model
            audio_instance.transcription = transcription
            # Detect keywords in the full text
            keywords_list = [kw.strip() for kw in new_keywords.split(",")]
            detected_keywords = detect_keywords(transcription_text, keywords_list)
            audio_instance.keywords_detected = list(detected_keywords)
            audio_instance.status = "reanalyzed"
            audio_instance.duration = len(transcription_text.split())
            audio_instance.save()
        except Exception as save_e:
            logger.error(f"Error saving transcription: {str(save_e)}")
            return Response({"error": "Failed to save transcription data."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        try:
            detected_keywords_list = ast.literal_eval(detected_keywords)
            if not isinstance(detected_keywords_list, list):
                detected_keywords_list = []
        except (ValueError, SyntaxError):
            detected_keywords_list = []
        
        return Response(
            {
                "file_path": file_path,
                "transcription": transcription,
                "detected_keywords": detected_keywords_list,
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
    """
    Handles the creation of new Standard Operating Procedures (SOPs).
    Note: Consider merging into SOPListView/ViewSet if following strict RESTful patterns.
    """
    serializer_class = SOPSerializer
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Create SOP",
        description="Creates a new Standard Operating Procedure (SOP) with its steps. Requires 'operator' role.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=SOPSerializer,
        responses={
            201: SOPSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
        },
        tags=['SOP']
    )
    def post(self, request, token):
        """
        Creates a new SOP.
        The `created_by` field is automatically set to the authenticated user.
        Requires 'operator' role.
        """
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
    """
    Retrieves a list of Standard Operating Procedures (SOPs).
    Supports pagination and role-based filtering.
    """
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="List SOPs",
        description="Fetches a paginated list of SOPs. Includes the username of the creator along with the created_by field.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.'),
            OpenApiParameter('page', OpenApiTypes.INT, OpenApiParameter.QUERY, description='Page number for pagination.'),
        ],
        responses={
            200: SOPSerializer(many=True),  # Actual response is paginated
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
        },
        tags=['SOP']
    )
    def get(self, request, token, format=None):
        """
        Lists SOPs with pagination.
        Includes the username of the creator in the response.
        """
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        try:
            if user_data['user'].role == 'admin':
                sops = SOP.objects.all().select_related('created_by').order_by('-created_at')
            else:
                # For non-admin users, filter SOPs based on their role
                sops = SOP.objects.filter(created_by=user_data['user']).select_related('created_by').order_by('-created_at')
            # sops = SOP.objects.all().select_related('created_by').order_by('-created_at')  # Use select_related for optimization
            paginator = PageNumberPagination()
            paginated_queryset = paginator.paginate_queryset(sops, request, view=self)

            # Add username to the response
            response_data = [
                {
                    **SOPSerializer(sop).data,
                    "created_by_username": sop.created_by.username if sop.created_by else None
                }
                for sop in paginated_queryset
            ]

            return paginator.get_paginated_response(response_data)
        except Exception as e:
            logger.error(f"Error fetching SOPs with pagination: {str(e)}")
            return Response({"error": f"Error fetching SOPs: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
class SOPDetailView(APIView):
    """
    Handles Retrieve, Update, and Delete operations for a specific SOP.
    """
    permission_classes = [RoleBasedPermission]
    # serializer_class = SOPSerializer # For drf-spectacular to pick up if not specified in extend_schema

    def get_object(self, sop_id):
        """Helper method to get SOP object by ID."""
        try:
            return SOP.objects.get(pk=sop_id)
        except SOP.DoesNotExist:
            return None

    @extend_schema(
        summary="Retrieve SOP Details",
        description="Fetches the details of a specific Standard Operating Procedure (SOP) by its ID.",
        parameters=[
            OpenApiParameter('sop_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the SOP to retrieve.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            200: SOPSerializer,
            404: ErrorResponseSerializer,
            401: ErrorResponseSerializer, # Assuming RoleBasedPermission handles this
        },
        tags=['SOP']
    )
    def get(self, request, sop_id, token):
        """
        Retrieve a specific SOP by its ID.
        Accessible by any authenticated user whose role allows access via RoleBasedPermission (admin, operator, reviewer).
        """
        # Token validation is handled by RoleBasedPermission
        # request.validated_user is set by RoleBasedPermission
        sop = self.get_object(sop_id)
        if sop is None:
            return Response({"error": "SOP not found"}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = SOPSerializer(sop)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @extend_schema(
        summary="Update SOP",
        description="Updates an existing Standard Operating Procedure (SOP). Requires 'admin' role or SOP ownership (user who created the SOP).",
        parameters=[
            OpenApiParameter('sop_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the SOP to update.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=SOPSerializer,
        responses={
            200: SOPSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['SOP']
    )
    def put(self, request, sop_id, token):
        """
        Updates an SOP.
        Requires 'admin' role or the user to be the creator of the SOP.
        """
        # request.validated_user is set by RoleBasedPermission
        sop = self.get_object(sop_id)
        if sop is None:
            return Response({"error": "SOP not found"}, status=status.HTTP_404_NOT_FOUND)

        if not (request.validated_user.role == 'admin' or sop.created_by == request.validated_user):
            logger.warning(f"Permission denied for user {request.validated_user.username} to update SOP {sop_id}")
            return Response({"error": "You do not have permission to perform this action."}, status=status.HTTP_403_FORBIDDEN)

        serializer = SOPSerializer(sop, data=request.data, partial=False) 
        if serializer.is_valid():
            updated_sop = serializer.save()
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

    @extend_schema(
        summary="Delete SOP",
        description="Deletes an existing Standard Operating Procedure (SOP). Requires 'admin' role or SOP ownership.",
        parameters=[
            OpenApiParameter('sop_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the SOP to delete.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            204: OpenApiTypes.NONE, 
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['SOP']
    )
    def delete(self, request, sop_id, token):
        """
        Deletes an SOP.
        Requires 'admin' role or the user to be the creator of the SOP.
        """
        # request.validated_user is set by RoleBasedPermission
        sop = self.get_object(sop_id)
        if sop is None:
            return Response({"error": "SOP not found"}, status=status.HTTP_404_NOT_FOUND)

        if not (request.validated_user.role == 'admin' or sop.created_by == request.validated_user):
            logger.warning(f"Permission denied for user {request.validated_user.username} to delete SOP {sop_id}")
            return Response({"error": "You do not have permission to perform this action."}, status=status.HTTP_403_FORBIDDEN)

        sop_name = sop.name 
        sop.delete()
        AuditLog.objects.create(
            action='sop_delete', 
            user=request.validated_user,
            object_id=sop_id, 
            object_type='SOP',
            details={'name': sop_name}
        )
        logger.info(f"SOP {sop_id} deleted by user {request.validated_user.username}")
        return Response(status=status.HTTP_204_NO_CONTENT)
    
class SessionCreateView(CreateAPIView):
    """
    Handles the creation of new Sessions.
    Note: Consider merging into SessionListView/ViewSet if following strict RESTful patterns.
    """
    serializer_class = SessionSerializer
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Create Session",
        description="Creates a new Session. Requires 'operator' role.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=SessionSerializer,
        responses={
            201: SessionSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
        },
        tags=['Session']
    )
    def post(self, request, token, format=None):
        """
        Creates a new session.
        The `user` (owner) field is automatically set to the authenticated user.
        Requires 'operator' role.
        """
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.serializer_class(data=request.data)
        # print(f"Request data: {request.data}") # Debug
        # print(f"User data: {user_data}") # Debug
        if serializer.is_valid():
            serializer.save(user=user_data['user'])
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SessionListView(APIView):
    """
    Retrieves a list of Sessions.
    Supports pagination and role-based filtering.
    """
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="List Sessions",
        description="Fetches a paginated list of Sessions. Admins see all sessions. Operators see sessions they created.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.'),
            OpenApiParameter('page', OpenApiTypes.INT, OpenApiParameter.QUERY, description='Page number for pagination.'),
        ],
        responses={
            200: SessionSerializer(many=True), # Actual response is paginated
            401: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=['Session']
    )
    def get(self, request, token, format=None):
        """
        Lists sessions with pagination.
        - Admins: Can see all sessions.
        - Operators: Can see sessions they created.
        - Reviewers: Currently not granted direct access to this list view by default (permission would deny).
        """
        user_data = token_verification(token)
        if user_data['status'] != 200: # Should be caught by permission class
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        try:
            if user_data['user'].role == 'admin':
                sessions = Session.objects.all()
            else:
                sessions = Session.objects.filter(user=user_data['user'])
            sessions = sessions.order_by("-created_at")
            
            paginator = PageNumberPagination()
            paginated_queryset = paginator.paginate_queryset(sessions, request, view=self)
            serializer = SessionSerializer(paginated_queryset, many=True)
            return paginator.get_paginated_response(serializer.data)
        except Exception as e:
            logger.error(f"Error fetching sessions with pagination: {str(e)}")
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
            
        paginator = PageNumberPagination()
        paginated_queryset = paginator.paginate_queryset(queryset, request, view=self)
        serializer = FeedbackReviewSerializer(paginated_queryset, many=True)
        return paginator.get_paginated_response(serializer.data)

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

class AdminDashboardSummaryView(APIView):
    permission_classes = [RoleBasedPermission] 

    @extend_schema(
        summary="Admin Dashboard Summary",
        description="Provides a summary of key metrics for the admin dashboard. Requires 'admin' role.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            200: OpenApiTypes.OBJECT, # Define a serializer if structure is fixed and complex
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=['Admin']
    )
    def get(self, request, token):
        """
        Retrieves aggregated data for the admin dashboard.
        - Total users
        - Total SOPs
        - Active sessions
        - Pending reviews
        Requires 'admin' role.
        """
        if request.validated_user.role != 'admin':
            return Response({"error": "Forbidden. Admin access required."}, status=status.HTTP_403_FORBIDDEN)

        try:
            total_users = UserProfile.objects.count()
            total_sops = SOP.objects.count()
            active_sessions = Session.objects.filter(status='active').count()
            pending_reviews = FeedbackReview.objects.filter(resolved_flag=False).count()

            data = {
                "total_users": total_users,
                "total_sops": total_sops,
                "active_sessions": active_sessions,
                "pending_reviews": pending_reviews
            }
            return Response(data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error generating admin dashboard summary: {str(e)}")
            return Response({"error": "An error occurred while generating the summary."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AdminUserListView(APIView):
    """
    Admin view to list all user profiles.
    Supports pagination.
    """
    permission_classes = [RoleBasedPermission] 

    @extend_schema(
        summary="Admin: List Users",
        description="Retrieves a paginated list of all user profiles. Requires 'admin' role.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.'),
            OpenApiParameter('page', OpenApiTypes.INT, OpenApiParameter.QUERY, description='Page number for pagination.'),
        ],
        responses={
            200: AdminUserProfileSerializer(many=True), # Actual response is paginated
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
        },
        tags=['Admin User Management']
    )
    def get(self, request, token):
        """
        Lists all user profiles with pagination.
        Requires 'admin' role.
        """
        if request.validated_user.role != 'admin':
            return Response({"error": "Forbidden. Admin access required."}, status=status.HTTP_403_FORBIDDEN)

        users = UserProfile.objects.all().order_by('id')
        
        paginator = PageNumberPagination()
        paginated_queryset = paginator.paginate_queryset(users, request, view=self)
        serializer = AdminUserProfileSerializer(paginated_queryset, many=True)
        return paginator.get_paginated_response(serializer.data)

class AdminUserDetailView(APIView):
    """
    Admin view to retrieve, update, and delete a specific user profile.
    """
    permission_classes = [RoleBasedPermission] 

    def get_object(self, user_id):
        """Helper method to get UserProfile object by ID."""
        try:
            return UserProfile.objects.get(pk=user_id)
        except UserProfile.DoesNotExist:
            return None

    @extend_schema(
        summary="Admin: Retrieve User Details",
        description="Fetches the details of a specific user profile by their ID. Requires 'admin' role.",
        parameters=[
            OpenApiParameter('user_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the user to retrieve.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            200: AdminUserProfileSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['Admin User Management']
    )
    def get(self, request, user_id, token):
        """
        Retrieve a specific user profile by ID.
        Requires 'admin' role.
        """
        if request.validated_user.role != 'admin':
            return Response({"error": "Forbidden. Admin access required."}, status=status.HTTP_403_FORBIDDEN)
        
        user_profile = self.get_object(user_id)
        if user_profile is None:
            return Response({"error": "UserProfile not found."}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = AdminUserProfileSerializer(user_profile)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @extend_schema(
        summary="Admin: Update User",
        description="Updates an existing user profile. Requires 'admin' role. Admins cannot change their own role or active status via this endpoint.",
        parameters=[
            OpenApiParameter('user_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the user to update.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=AdminUserProfileSerializer,
        responses={
            200: AdminUserProfileSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['Admin User Management']
    )
    def put(self, request, user_id, token):
        """Handles full updates for a user profile by an admin."""
        return self._update_handler(request, user_id, token, partial=False)

    @extend_schema(
        summary="Admin: Partially Update User",
        description="Partially updates an existing user profile. Requires 'admin' role. Admins cannot change their own role or active status via this endpoint.",
        parameters=[
            OpenApiParameter('user_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the user to partially update.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=AdminUserProfileSerializer,
        responses={
            200: AdminUserProfileSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['Admin User Management']
    )
    def patch(self, request, user_id, token):
        """Handles partial updates for a user profile by an admin."""
        return self._update_handler(request, user_id, token, partial=True)

    # _update_handler method already exists

    @extend_schema(
        summary="Admin: Delete User",
        description="Deletes an existing user profile. Requires 'admin' role. Admins cannot delete themselves via this endpoint.",
        parameters=[
            OpenApiParameter('user_id', OpenApiTypes.INT, OpenApiParameter.PATH, description='The ID of the user to delete.'),
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            204: OpenApiTypes.NONE,
            400: ErrorResponseSerializer, # For trying to delete self
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=['Admin User Management']
    )
    def delete(self, request, user_id, token):
        """
        Deletes a user profile.
        Requires 'admin' role. Admins cannot delete their own account via this API.
        """
        if request.validated_user.role != 'admin':
            return Response({"error": "Forbidden. Admin access required."}, status=status.HTTP_403_FORBIDDEN)

        user_profile_to_delete = self.get_object(user_id)
        if user_profile_to_delete is None:
            return Response({"error": "UserProfile not found."}, status=status.HTTP_404_NOT_FOUND)

        if user_profile_to_delete == request.validated_user:
            logger.warning(f"Admin user {request.validated_user.username} attempted to delete themselves via API.")
            return Response({"error": "Admin users cannot delete themselves via the API."}, status=status.HTTP_400_BAD_REQUEST)

        username_for_log = user_profile_to_delete.username
        user_profile_to_delete.delete()

        AuditLog.objects.create(
            action='userprofile_delete',
            user=request.validated_user, 
            object_id=user_id, 
            object_type='UserProfile',
            details={'deleted_user_id': user_id, 'deleted_username': username_for_log}
        )
        logger.info(f"UserProfile {user_id} ({username_for_log}) deleted by admin {request.validated_user.username}")
        return Response(status=status.HTTP_204_NO_CONTENT)

# def transcribe_with_speaker_diarization(audio_url: str, model_path: str, speaker_model_path: str):
#     """
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
        
        paginator = PageNumberPagination()
        paginated_queryset = paginator.paginate_queryset(users, request, view=self)
        # AdminUserProfileSerializer is already imported in the original file if this view exists
        serializer = AdminUserProfileSerializer(paginated_queryset, many=True)
        return paginator.get_paginated_response(serializer.data)

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
    """
    Manages user-specific settings.
    Allows users to retrieve and update their own settings.
    """
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Get User Settings",
        description="Retrieves the settings for the authenticated user.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            200: UserSettingsSerializer,
            401: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=['User Settings']
    )
    def get(self, request, token, format=None):
        """
        Retrieves settings for the authenticated user.
        Accessible by any authenticated user for their own settings.
        """
        logger.info(f"Fetching user settings for user: {request.validated_user.username}")
        try:
            settings, created = UserSettings.objects.get_or_create(user=request.validated_user)
            serializer = UserSettingsSerializer(settings)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching user settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @extend_schema(
        summary="Update User Settings",
        description="Partially updates settings for the authenticated user. Allows changing language, notification preferences, and theme.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=UserSettingsSerializer, # Or a more specific one if some fields are not updatable
        responses={
            200: UserSettingsSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=['User Settings']
    )
    def patch(self, request, token, format=None):
        """
        Partially updates settings for the authenticated user.
        Also updates UserProfile.theme if 'theme' is in request.data.
        Accessible by any authenticated user for their own settings.
        """
        logger.info(f"Updating user settings for user: {request.validated_user.username}")
        try:
            settings, created = UserSettings.objects.get_or_create(user=request.validated_user)
            serializer = UserSettingsSerializer(settings, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                if 'theme' in request.data:
                    request.validated_user.theme = request.data['theme']
                    request.validated_user.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error updating user settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SystemSettingsView(APIView):
    """
    Manages system-wide settings.
    Accessible only by Admin users.
    """
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Get System Settings",
        description="Retrieves system-wide settings. Requires 'admin' role.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        responses={
            200: SystemSettingsSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer, # If non-admin tries to access
            500: ErrorResponseSerializer,
        },
        tags=['System Settings']
    )
    def get(self, request, token, format=None):
        """
        Retrieves system settings.
        Requires 'admin' role.
        """
        logger.info(f"Fetching system settings for user: {request.validated_user.username}")
        try:
            settings, created = SystemSettings.objects.get_or_create(id=1) # Assuming singleton
            serializer = SystemSettingsSerializer(settings)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching system settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @extend_schema(
        summary="Update System Settings",
        description="Partially updates system-wide settings. Requires 'admin' role.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.')
        ],
        request=SystemSettingsSerializer,
        responses={
            200: SystemSettingsSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer, # If non-admin tries to access
            500: ErrorResponseSerializer,
        },
        tags=['System Settings']
    )
    def patch(self, request, token, format=None):
        """
        Partially updates system settings.
        Requires 'admin' role.
        """
        logger.info(f"Updating system settings for user: {request.validated_user.username}")
        try:
            settings, created = SystemSettings.objects.get_or_create(id=1) # Assuming singleton
            serializer = SystemSettingsSerializer(settings, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error updating system settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AuditLogView(APIView):
    """
    Retrieves a list of audit log entries.
    Supports pagination. Access restricted by role.
    """
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="List Audit Logs",
        description="Fetches a paginated list of audit log entries. Access restricted to 'admin' and 'reviewer' roles.",
        parameters=[
            OpenApiParameter('token', OpenApiTypes.STR, OpenApiParameter.PATH, description='Authentication token.'),
            OpenApiParameter('page', OpenApiTypes.INT, OpenApiParameter.QUERY, description='Page number for pagination.'),
        ],
        responses={
            200: AuditLogSerializer(many=True), # Actual response is paginated
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer, # If operator tries to access
            500: ErrorResponseSerializer,
        },
        tags=['Audit Log']
    )
    def get(self, request, token, format=None):
        """
        Lists audit logs with pagination.
        Accessible by 'admin' and 'reviewer' roles.
        """
        logger.info(f"Fetching audit logs for user: {request.validated_user.username}")
        try:
            # Permission class should already restrict this, but double-check
            if request.validated_user.role not in ['admin', 'reviewer']:
                 return Response({"error": "Forbidden. You do not have permission to view audit logs."}, status=status.HTTP_403_FORBIDDEN)

            logs = AuditLog.objects.all().order_by('-timestamp')
            
            paginator = PageNumberPagination()
            paginated_queryset = paginator.paginate_queryset(logs, request, view=self)
            serializer = AuditLogSerializer(paginated_queryset, many=True)
            return paginator.get_paginated_response(serializer.data)
        except Exception as e:
            logger.error(f"Error fetching audit logs with pagination: {str(e)}")
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
