from rest_framework import serializers
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from .models import ReferenceDocument, AudioFile, ProcessingSession, UserProfile
from .new_utils import allowed_file
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist


class UploadAndProcessSerializer(serializers.Serializer):
    """
    Serializer for the main processing endpoint.
    Validates that either a new text file is uploaded or an existing document ID is provided.
    """
    # --- MODIFIED: text_file is no longer always required ---
    text_file = serializers.FileField(
        help_text="Upload a new text document (PDF, DOCX, or TXT). Required if 'existing_document_id' is not provided.",
        required=False # Changed from True
    )
    
    # --- NEW: Field for existing document ID ---
    existing_document_id = serializers.UUIDField(
        help_text="ID of an existing reference document to use. Required if 'text_file' is not provided.",
        required=False
    )

    # --- UNCHANGED: audio_file is always required ---
    audio_file = serializers.FileField(
        help_text="Upload an audio file (MP3, WAV, M4A, MPEG, MP4)",
        required=True
    )
    
    document_type = serializers.ChoiceField(
        choices=ReferenceDocument.DOCUMENT_TYPES,
        default='sop',
        required=False,
        help_text="Type of the document, used only when uploading a new 'text_file'."
    )
    document_name = serializers.CharField(
        max_length=255,
        required=False,
        allow_blank=True,
        help_text="Custom name for the document, used only when uploading a new 'text_file'."
    )

    def validate(self, data):
        """
        Cross-field validation to ensure either text_file or existing_document_id is provided.
        """
        text_file = data.get('text_file')
        doc_id = data.get('existing_document_id')

        # Case 1: Neither is provided
        if not text_file and not doc_id:
            raise serializers.ValidationError(
                "You must provide either a 'text_file' to upload or an 'existing_document_id' of a document to reuse."
            )
        
        # Case 2: Both are provided
        if text_file and doc_id:
            raise serializers.ValidationError(
                "Please provide either 'text_file' or 'existing_document_id', but not both."
            )
            
        # Case 3: ID is provided, let's check if it's valid
        if doc_id:
            try:
                ReferenceDocument.objects.get(id=doc_id)
            except ObjectDoesNotExist:
                raise serializers.ValidationError({
                    "existing_document_id": f"No ReferenceDocument found with the ID '{doc_id}'."
                })

        return data

    def validate_text_file(self, value):
        """Validate text file extension and size (unchanged)"""
        if not allowed_file(value.name, settings.ALLOWED_TEXT_EXTENSIONS):
            raise serializers.ValidationError(
                "Invalid file type. Please upload a PDF, DOCX, or TXT file."
            )
        if value.size > 50 * 1024 * 1024:
            raise serializers.ValidationError(
                "File size too large. Maximum size is 50MB."
            )
        return value

    def validate_audio_file(self, value):
        """Validate audio file extension and size (unchanged)"""
        if not allowed_file(value.name, settings.ALLOWED_AUDIO_EXTENSIONS):
            raise serializers.ValidationError(
                "Invalid file type. Please upload an MP3, WAV, M4A, MPEG, or MP4 file."
            )
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
    session_id = serializers.CharField(
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


class RunDiarizationSerializer(serializers.Serializer):
    audio_id = serializers.UUIDField()


class SpeakerProfileMappingEntrySerializer(serializers.Serializer):
    label = serializers.CharField()
    name = serializers.CharField()
    profile_id = serializers.IntegerField(required=False, allow_null=True)

    def validate_name(self, value: str) -> str:
        if not value.strip():
            raise serializers.ValidationError("Name cannot be blank.")
        return value


class SpeakerProfileMappingSerializer(serializers.Serializer):
    audio_id = serializers.UUIDField()
    speakers = SpeakerProfileMappingEntrySerializer(many=True)

    def validate_speakers(self, value):
        labels = set()
        for entry in value:
            label = entry.get('label')
            if not label:
                raise serializers.ValidationError("Each speaker entry must include a label.")
            if label in labels:
                raise serializers.ValidationError("Duplicate speaker labels are not allowed.")
            labels.add(label)
        return value

class ReferenceDocumentSerializer(serializers.Serializer):
    """
    Serializer for the ReferenceDocument model.
    Used to validate input and format the output.
    """
    file_path = serializers.FileField(
        help_text="Upload a text document (PDF, DOCX, or TXT)",
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
    # class Meta:
    #     model = ReferenceDocument
    #     fields = [
    #         'id', 
    #         'name', 
    #         'document_type', 
    #         'file_path', 
    #         'original_filename', 
    #         'file_size', 
    #         'content_type',
    #         'upload_status', 
    #         'uploaded_by', 
    #         'created_at',
    #         'extracted_text' # Included for response, but not for input
    #     ]
    #     # These fields are set by the server, not provided by the client on upload.
    #     read_only_fields = [
    #         'id', 
    #         'file_path', 
    #         'original_filename', 
    #         'file_size', 
    #         'content_type', 
    #         'upload_status',
    #         'uploaded_by', 
    #         'created_at', 
    #         'extracted_text'
    #     ]