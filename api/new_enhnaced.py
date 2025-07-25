import os
import tempfile
import uuid
import time
from datetime import datetime, timedelta
from django.conf import settings
from django.http import HttpResponse, Http404
from django.utils import timezone
from rest_framework import status
from rest_framework.generics import CreateAPIView, GenericAPIView
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import traceback
from .models import ReferenceDocument, AudioFile, ProcessingSession, UserProfile, AuditLog
from .new_utils import (
    extract_text_from_s3, transcribe_audio_from_s3, diarization_from_audio,  # <-- add diarization_from_audio
    find_missing, create_highlighted_docx_from_s3, upload_file_to_s3,
    get_s3_key_from_url, s3_client
)
from .authentication import token_verification
from .new_serializers import (
    UploadAndProcessSerializer, ProcessingResultSerializer,
    DownloadRequestSerializer, UserDocumentsSerializer,
    CleanupRequestSerializer, CleanupResponseSerializer,
    ErrorResponseSerializer, ProcessingSessionDetailSerializer,
    ReferenceDocumentSerializer
)
import requests
from pyannote.audio import Pipeline
import subprocess
import threading

class UploadAndProcessView(CreateAPIView):
    """
    Upload text and audio files to S3, process them, and return analysis results
    """
    serializer_class = UploadAndProcessSerializer
    #authentication_classes = [KnoxTokenAuthentication]
    
    @swagger_auto_schema(
        operation_description="Upload and process text document with audio file",
        responses={
            200: ProcessingResultSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            500: ErrorResponseSerializer
        }
    )
    def create(self, request, token, *args, **kwargs):
        # Verify token manually since authentication might not work with URL token
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            error_data = {
                'error': auth_result['error'],
                'timestamp': timezone.now()
            }
            error_serializer = ErrorResponseSerializer(data=error_data)
            error_serializer.is_valid()
            return Response(error_data, status=status.HTTP_401_UNAUTHORIZED)
        
        user = auth_result['user']
        
        # Validate input data
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            error_data = {
                'error': 'Invalid input data',
                'details': serializer.errors,
                'timestamp': timezone.now()
            }
            error_serializer = ErrorResponseSerializer(data=error_data)
            error_serializer.is_valid()
            return Response(error_data, status=status.HTTP_400_BAD_REQUEST)
        
        start_time = time.time()
        reference_doc = None
        audio_obj = None
        
        try:
            validated_data = serializer.validated_data
            user_profile = user
            audio_file = validated_data['audio_file']

            # --- MODIFIED LOGIC: Handle document creation/fetching ---
            
            if 'text_file' in validated_data:
                # --- PATH 1: A new text file was uploaded ---
                text_file = validated_data['text_file']
                document_type = validated_data.get('document_type', 'sop')
                document_name = validated_data.get('document_name', '')

                # Upload text file to S3
                text_s3_key = f'documents/{user.id}/{uuid.uuid4()}_{text_file.name}'
                text_s3_url = upload_file_to_s3(text_file, text_s3_key)
                
                # Create ReferenceDocument
                reference_doc = ReferenceDocument.objects.create(
                    name=document_name or text_file.name,
                    document_type=document_type,
                    file_path=text_s3_url,
                    original_filename=text_file.name,
                    file_size=text_file.size,
                    content_type=text_file.content_type or 'application/octet-stream',
                    uploaded_by=user_profile,
                    upload_status='processing'
                )
                
                # Extract text from the newly uploaded document
                text_content = extract_text_from_s3(text_s3_url)
                reference_doc.extracted_text = text_content
                reference_doc.upload_status = 'processed'
                reference_doc.save()
            
            else:
                # --- PATH 2: An existing document ID was provided ---
                doc_id = validated_data['existing_document_id']
                reference_doc = ReferenceDocument.objects.get(id=doc_id)
                
                # Optional: Ensure text is extracted if it was missed before
                if not reference_doc.extracted_text and reference_doc.file_path:
                    print(f"Extracting missing text for existing document: {reference_doc.id}")
                    text_content = extract_text_from_s3(reference_doc.file_path)
                    reference_doc.extracted_text = text_content
                    reference_doc.save(update_fields=['extracted_text'])
            
            # Upload audio file to S3
            audio_s3_key = f'audio/{user.id}/{uuid.uuid4()}_{audio_file.name}'
            audio_s3_url = upload_file_to_s3(audio_file, audio_s3_key)
            
            # Create AudioFile
            audio_obj = AudioFile.objects.create(
                file_path=audio_s3_url,
                original_filename=audio_file.name,
                user=user_profile,
                reference_document=reference_doc,
                status='processing'
            )
            
            # # Extract text from document
            text_content = extract_text_from_s3(reference_doc.file_path)
            reference_doc.extracted_text = text_content
            reference_doc.upload_status = 'processed'
            reference_doc.save()
            
            # Transcribe audio
            transcript_result = transcribe_audio_from_s3(audio_s3_url)
            transcript = transcript_result["text"]
            transcript_segments = transcript_result.get("segments", [])
            transcript_words = transcript_result.get("words", [])
            audio_obj.transcription = transcript_result
            audio_obj.save()

            # --- Speaker Diarization (using diarization_from_audio) ---
            diarization_segments = []
            diarization_error = None
            # try:
            #     diarization_segments = diarization_from_audio(audio_s3_url, transcript_segments, transcript_words)
            # except Exception as diar_err:
            #     diarization_error = str(diar_err)
            #     print(f"Diarization error: {diarization_error}")
            # audio_obj.diarization = diarization_segments if diarization_segments else None
            # audio_obj.save()
            # --- End Speaker Diarization ---
            
            # --- Speaker Diarization in Background Thread ---
            def diarization_background(audio_file_id, audio_s3_url, transcript_segments, transcript_words):
                try:
                    diarization_segments = diarization_from_audio(audio_s3_url, transcript_segments, transcript_words)
                    audio_file = AudioFile.objects.get(id=audio_file_id)
                    audio_file.diarization = diarization_segments
                    audio_file.save()
                except Exception as diar_err:
                    audio_file = AudioFile.objects.get(id=audio_file_id)
                    audio_file.diarization = None
                    audio_file.save()
                    print(f"Diarization error (background): {str(diar_err)}")

            diarization_thread = threading.Thread(
                target=diarization_background,
                args=(audio_obj.id, audio_s3_url, transcript_segments, transcript_words),
                daemon=True
            )
            diarization_thread.start()
            # --- End Speaker Diarization ---
            
            # Perform comparison analysis
            matched_html, missing_html, matched_words, total_words, entire_html = find_missing(
                text_content, transcript
            )
            
            # Calculate coverage
            coverage = (matched_words / total_words * 100) if total_words > 0 else 0
            
            # Update audio object with results
            audio_obj.coverage = coverage
            audio_obj.status = 'processed'
            audio_obj.save()
            
            # Create processing session for download
            expires_at = timezone.now() + timedelta(hours=24)
            session = ProcessingSession.objects.create(
                reference_document=reference_doc,
                audio_file=audio_obj,
                matched_words=matched_words,
                total_words=total_words,
                coverage=coverage,
                expires_at=expires_at
            )
            
            processing_time = time.time() - start_time

            audio_obj.processing_session = session.id
            audio_obj.save()
            
            # Prepare response data - include diarization
            response_data = {
                'session_id': str(session.id),
                'matched_words': matched_words,
                'total_words': total_words,
                'coverage': round(coverage, 2),
                'reference_document_id': str(reference_doc.id),
                'audio_file_id': str(audio_obj.id),
                'matched_content': matched_html,
                'missing_content': missing_html,
                'entire_document': entire_html,
                'processing_time': round(processing_time, 2),
                'diarization': diarization_segments if diarization_segments else None,
                'diarization_error': diarization_error
            }

            # Don't validate response data with serializer, just return it directly
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            # Clean up on error
            if reference_doc:
                reference_doc.upload_status = 'failed'
                reference_doc.save()
            if audio_obj:
                audio_obj.status = 'failed'
                audio_obj.save()
            
            # Log the error for debugging
            import traceback
            print(f"Error in UploadAndProcessView: {str(e)}")
            print(traceback.format_exc())
            
            error_data = {
                'error': f'Processing failed: {str(e)}',
                'timestamp': timezone.now().isoformat()  # Convert datetime to string
            }
            
            return Response(error_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
                id=session_id,
                expires_at__gt=timezone.now()
            )
            print(f"Session found: {session.id}")
            
            # Verify user has access
            # if (session.audio_file.user != user or session.audio_file.user.role != 'reviewer'):
            #     return Response({
            #         'error': 'Access denied - You can only access your own sessions',
            #         'timestamp': timezone.now().isoformat()
            #     }, status=status.HTTP_403_FORBIDDEN)
            
            reference_doc = session.reference_document
            audio_file = session.audio_file
            
            print(f"Reference doc: {reference_doc.name}")
            print(f"Audio file: {audio_file.original_filename}")
            
            # Check if processed DOCX already exists in S3
            if session.processed_docx_path:
                print(f"Using existing processed document: {session.processed_docx_path}")
                processed_s3_url = session.processed_docx_path
            else:
                print("Creating new processed document...")
                try:
                    # Create highlighted DOCX and upload to S3
                    transcript = audio_file.transcription.get('text', '') if audio_file.transcription else ''
                    
                    if not transcript:
                        return Response({
                            'error': 'No transcript available for processing',
                            'timestamp': timezone.now().isoformat()
                        }, status=status.HTTP_400_BAD_REQUEST)
                    
                    print(f"Transcript length: {len(transcript)} characters")
                    print(f"Reference doc path: {reference_doc.file_path}")
                    
                    # Create highlighted document
                    processed_s3_url = create_highlighted_docx_from_s3(
                        reference_doc.file_path, 
                        transcript
                    )
                    
                    print(f"Created processed document: {processed_s3_url}")
                    
                    # Save S3 URL to session
                    session.processed_docx_path = processed_s3_url
                    session.save()
                    
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

            # AuditLog.objects.create(
            #     action='Download Report',
            #     user=user_data['user'],
            #     # user = user_data['user'],
            #     # session_id=session_id,
            #     object_id=session.id,
            #     object_type='AudioFile',
            #     details={
            #         "session_id": session.id
            #             # 'file_name': audio_file.name,
            #             # 'file_size': audio_file.size,
            #             # 'speaker_stats': speaker_stats,
            #             # 'processing_method': 'enhanced_diarization'
            #         }
            # )
            
            # Option 1: Return S3 URL (Current implementation)
            return Response({
                'processed_docx_url': processed_s3_url,
                'message': 'Processed document is available at the above URL.',
                'session_id': str(session.id),
                'filename': f"{reference_doc.original_filename.rsplit('.', 1)[0]}_processed.docx"
            }, status=status.HTTP_200_OK)
            
            # Option 2: Direct file download (Uncomment if you want direct download)
            """
            try:
                s3_key = get_s3_key_from_url(processed_s3_url)
                temp_path = download_file_from_s3(s3_key)
                
                with open(temp_path, 'rb') as f:
                    response = HttpResponse(
                        f.read(),
                        content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    )
                    download_name = f"{reference_doc.original_filename.rsplit('.', 1)[0]}_processed.docx"
                    response['Content-Disposition'] = f'attachment; filename="{download_name}"'
                    
                # Clean up temp file
                os.unlink(temp_path)
                return response
                
            except Exception as download_error:
                print(f"Error downloading from S3: {str(download_error)}")
                return Response({
                    'error': f'Failed to download processed document: {str(download_error)}',
                    'processed_docx_url': processed_s3_url,  # Provide URL as fallback
                    'timestamp': timezone.now().isoformat()
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            """
        
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
            
            serializer = self.get_serializer(response_data)
            return Response(serializer.data, status=status.HTTP_200_OK)
            
        except UserProfile.DoesNotExist:
            response_data = {
                'documents': [],
                'audio_files': [],
                'total_documents': 0,
                'total_audio_files': 0
            }
            serializer = self.get_serializer(response_data)
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
        
        serializer = self.get_serializer(data=request.data)
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
            # Delete processed DOCX file from S3
            if session.processed_docx_path:
                try:
                    s3_key = get_s3_key_from_url(session.processed_docx_path)
                    s3_client.delete_object(
                        Bucket=settings.AWS_STORAGE_BUCKET_NAME,
                        Key=s3_key
                    )
                    cleaned_files += 1
                except Exception as e:
                    print(f"Failed to delete S3 object {s3_key}: {str(e)}")
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
            
            serializer = self.get_serializer(session)
            return Response(serializer.data, status=status.HTTP_200_OK)
            
        except ProcessingSession.DoesNotExist:
            error_data = {
                'error': 'Processing session not found',
                'timestamp': timezone.now()
            }
            return Response(error_data, status=status.HTTP_404_NOT_FOUND)

class UploadReferenceDocumentView(CreateAPIView):
    """
    Upload Reference Document files to S3, process them and return analysis results
    """
    serializer_class = ReferenceDocumentSerializer

    @swagger_auto_schema(
        operation_description="Upload Reference Document files to S3, process them and return analysis results",
        responses={
            200: ReferenceDocumentSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            500: ErrorResponseSerializer
        }
    )
    def create(self, request, token, *args, **kwargs):
        # Verify token manually since authentication might not work with URL token
        auth_result = token_verification(token)
        if auth_result['status'] != 200:
            error_data = {
                'error': auth_result['error'],
                'timestamp': timezone.now()
            }
            return Response(error_data, status=status.HTTP_401_UNAUTHORIZED)
        
        user = auth_result['user']
        
        # Validate input data
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            error_data = {
                'error': 'Invalid input data',
                'details': serializer.errors,
                'timestamp': timezone.now()
            }
            return Response(error_data, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Get validated data
            text_file = serializer.validated_data['file_path']
            document_type = serializer.validated_data.get('document_type', 'sop')
            document_name = serializer.validated_data.get('document_name', '')
            
            # Since UserProfile extends AbstractUser, the user IS the UserProfile
            user_profile = user
            
            # Upload text file to S3
            text_s3_key = f'documents/{user.id}/{uuid.uuid4()}_{text_file.name}'
            text_s3_url = upload_file_to_s3(text_file, text_s3_key)
            
            # Create ReferenceDocument
            reference_doc = ReferenceDocument.objects.create(
                name=document_name or text_file.name,
                document_type=document_type,
                file_path=text_s3_url,
                original_filename=text_file.name,
                file_size=text_file.size,
                content_type=text_file.content_type or 'application/octet-stream',
                uploaded_by=user_profile,
                upload_status='processing'
            )
            # Extract text from document
            text_content = extract_text_from_s3(text_s3_url)
            reference_doc.extracted_text = text_content
            reference_doc.upload_status = 'processed'
            reference_doc.save()
            # AuditLog.objects.create(
            #     action='Upload Document',
            #     user=auth_result['user'],
            #     # object_id=str(reference_doc.id).replace('-',''),
            #     object_type='DocumentFile',
            #     details={
            #         "document_id":str(reference_doc.id).replace('-',''),
            #         "name":reference_doc.name,
            #         "size":reference_doc.file_size
            #         }
            # )
            return Response({"message":"Docuemnt Upload Success"}, status=status.HTTP_200_OK)    

        except Exception as e:
            print(f"Error in UploadAndProcessView: {str(e)}")
            print(traceback.format_exc())
            return Response({
                'error': f'Processing failed: {str(e)}',
                'timestamp': timezone.now().isoformat()  # Convert datetime to string
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)