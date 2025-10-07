from rest_framework import serializers
from django.core.exceptions import ObjectDoesNotExist
from .models import ReferenceDocument, AudioFile, ProcessingSession, UserProfile, RAGAssistant, RAGThread, RAGMessage, RAGRun
from .new_utils import allowed_file
from peercheck import settings


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
        text_file = data.get('text_file')
        doc_id = data.get('existing_document_id')

        # Ensure either text_file or existing_document_id is provided, not both
        if not text_file and not doc_id:
            raise serializers.ValidationError("Provide either 'text_file' or 'existing_document_id'.")
        if text_file and doc_id:
            raise serializers.ValidationError("Provide only one of 'text_file' or 'existing_document_id'.")
        
        # If existing_document_id is provided, ensure the document exists
        if doc_id:
            try:
                ReferenceDocument.objects.get(id=doc_id)
            except ObjectDoesNotExist:
                raise serializers.ValidationError({"existing_document_id": f"No ReferenceDocument with ID '{doc_id}'."})

        return data

    def validate_text_file(self, value):
        if not allowed_file(value.name, settings.ALLOWED_TEXT_EXTENSIONS):
            raise serializers.ValidationError("Invalid file type. Upload PDF, DOCX, or TXT.")
        if value.size > 50 * 1024 * 1024:
            raise serializers.ValidationError("Max file size is 50MB.")
        return value

    def validate_audio_file(self, value):
        if not allowed_file(value.name, settings.ALLOWED_AUDIO_EXTENSIONS):
            raise serializers.ValidationError("Invalid audio type. Upload MP3, WAV, M4A, MPEG, or MP4.")
        if value.size > 100 * 1024 * 1024:
            raise serializers.ValidationError("Max audio size is 100MB.")
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
    session_id = serializers.CharField(help_text="Processing session ID received from upload response")

class ReferenceDocumentDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceDocument
        fields = [
            'id', 'name', 'document_type', 'original_filename',
            'file_size', 'content_type', 'upload_status',
            'created_at', 'updated_at',
            # RAG fields
            'rag_enabled', 'rag_vector_store_id', 'rag_document_id',
            'rag_status', 'rag_uploaded_at', 'rag_ingested_at',
            'rag_last_error', 'rag_metadata'
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
    force_cleanup = serializers.BooleanField(default=False, help_text="Force cleanup of all expired sessions")

class CleanupResponseSerializer(serializers.Serializer):
    message = serializers.CharField(read_only=True)
    deleted_count = serializers.IntegerField(read_only=True)
    cleaned_files = serializers.IntegerField(read_only=True)

class ErrorResponseSerializer(serializers.Serializer):
    error = serializers.CharField(read_only=True)
    details = serializers.DictField(read_only=True, required=False)
    timestamp = serializers.DateTimeField(read_only=True)

class ReferenceDocumentSerializer(serializers.Serializer):
    file_path = serializers.FileField(required=True, help_text="Upload a text document (PDF, DOCX, or TXT)")
    document_type = serializers.ChoiceField(choices=ReferenceDocument.DOCUMENT_TYPES, default='sop', required=False)
    document_name = serializers.CharField(max_length=255, required=False, allow_blank=True)

class RunDiarizationSerializer(serializers.Serializer):
    audio_id = serializers.UUIDField()

class SpeakerProfileMappingSerializer(serializers.Serializer):
    audio_id = serializers.UUIDField()
    speaker_label = serializers.CharField(max_length=50)
    name = serializers.CharField(max_length=255)
    profile_id = serializers.IntegerField(required=False)

# --------- NEW: RAG conversational serializers (simple) ---------

class RAGAssistantSerializer(serializers.ModelSerializer):
    class Meta:
        model = RAGAssistant
        fields = ['external_id','name','model','vector_store_ids','created_at']

class RAGThreadSerializer(serializers.ModelSerializer):
    class Meta:
        model = RAGThread
        fields = ['external_id','assistant_external_id','title','created_at']

class RAGMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = RAGMessage
        fields = ['external_id','thread_external_id','role','content','created_at']

class RAGRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = RAGRun
        fields = ['external_id','thread_external_id','assistant_external_id','status','raw','created_at']
