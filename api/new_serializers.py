from rest_framework import serializers
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from .models import ReferenceDocument, AudioFile, ProcessingSession, UserProfile
from .new_utils import allowed_file
from django.conf import settings

class UploadAndProcessSerializer(serializers.Serializer):
    text_file = serializers.FileField(
        help_text="Upload a text document (PDF, DOCX, or TXT)",
        required=True
    )
    audio_file = serializers.FileField(
        help_text="Upload an audio file (MP3, WAV, M4A, MPEG, MP4)",
        required=True
    )
    document_type = serializers.ChoiceField(
        choices=ReferenceDocument.DOCUMENT_TYPES,
        default='sop',
        required=False,
        help_text="Type of the reference document"
    )
    document_name = serializers.CharField(
        max_length=255,
        required=False,
        allow_blank=True,
        help_text="Custom name for the document (optional)"
    )

    def validate_text_file(self, value):
        """Validate text file extension"""
        if not allowed_file(value.name, settings.ALLOWED_TEXT_EXTENSIONS):
            raise serializers.ValidationError(
                "Invalid file type. Please upload a PDF, DOCX, or TXT file."
            )
        
        # Check file size (e.g., max 50MB)
        if value.size > 50 * 1024 * 1024:
            raise serializers.ValidationError(
                "File size too large. Maximum size is 50MB."
            )
        
        return value

    def validate_audio_file(self, value):
        """Validate audio file extension"""
        if not allowed_file(value.name, settings.ALLOWED_AUDIO_EXTENSIONS):
            raise serializers.ValidationError(
                "Invalid file type. Please upload an MP3, WAV, M4A, MPEG, or MP4 file."
            )
        
        # Check file size (e.g., max 100MB)
        if value.size > 100 * 1024 * 1024:
            raise serializers.ValidationError(
                "File size too large. Maximum size is 100MB."
            )
        
        return value

class ProcessingResultSerializer(serializers.Serializer):
    session_id = serializers.UUIDField(read_only=True)
    matched_words = serializers.IntegerField(read_only=True)
    total_words = serializers.IntegerField(read_only=True)
    coverage = serializers.FloatField(read_only=True)
    reference_document_id = serializers.UUIDField(read_only=True)
    audio_file_id = serializers.UUIDField(read_only=True)
    matched_content = serializers.CharField(read_only=True)
    missing_content = serializers.CharField(read_only=True)
    entire_document = serializers.CharField(read_only=True)
    processing_time = serializers.FloatField(read_only=True, required=False)

class DownloadRequestSerializer(serializers.Serializer):
    session_id = serializers.UUIDField(
        help_text="Processing session ID received from upload response"
    )

class ReferenceDocumentDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceDocument
        fields = [
            'id', 'name', 'document_type', 'original_filename',
            'file_size', 'content_type', 'upload_status',
            'created_at', 'updated_at'
        ]

class AudioFileDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioFile
        fields = [
            'id', 'original_filename', 'status', 'duration',
            'coverage', 'created_at', 'updated_at'
        ]

class UserDocumentsSerializer(serializers.Serializer):
    documents = ReferenceDocumentDetailSerializer(many=True, read_only=True)
    audio_files = AudioFileDetailSerializer(many=True, read_only=True)
    total_documents = serializers.IntegerField(read_only=True)
    total_audio_files = serializers.IntegerField(read_only=True)

class ProcessingSessionDetailSerializer(serializers.ModelSerializer):
    reference_document = ReferenceDocumentDetailSerializer(read_only=True)
    audio_file = AudioFileDetailSerializer(read_only=True)
    
    class Meta:
        model = ProcessingSession
        fields = [
            'id', 'matched_words', 'total_words', 'coverage',
            'created_at', 'expires_at', 'reference_document', 'audio_file'
        ]

class CleanupRequestSerializer(serializers.Serializer):
    force_cleanup = serializers.BooleanField(
        default=False,
        help_text="Force cleanup of all expired sessions"
    )

class CleanupResponseSerializer(serializers.Serializer):
    message = serializers.CharField(read_only=True)
    deleted_count = serializers.IntegerField(read_only=True)
    cleaned_files = serializers.IntegerField(read_only=True)

class ErrorResponseSerializer(serializers.Serializer):
    error = serializers.CharField(read_only=True)
    details = serializers.DictField(read_only=True, required=False)
    timestamp = serializers.DateTimeField(read_only=True)
