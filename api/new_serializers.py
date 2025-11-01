import json

from rest_framework import serializers
from django.core.exceptions import ObjectDoesNotExist
from .models import (
    ReferenceDocument,
    AudioFile,
    ProcessingSession,
    UserProfile,
    SpeakerProfile,
    RAGAssistant,
    RAGThread,
    RAGMessage,
    RAGRun,
)
from .new_utils import allowed_file
from peercheck import settings


class UploadAndProcessSerializer(serializers.Serializer):
    """
    Serializer for the main processing endpoint.
    Callers may upload a new reference document, reuse an existing one, or rely
    entirely on automatic RAG matching by supplying only an audio file.
    """
    text_file = serializers.FileField(
        help_text="Upload a new text document (PDF, DOCX, or TXT). Optional.",
        required=False
    )

    existing_document_id = serializers.UUIDField(
        help_text="ID of an existing reference document to use. Optional.",
        required=False
    )

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

        # Ensure the client does not upload a new document while also pointing to an existing one
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
    diarization_status = serializers.CharField(read_only=True, required=False)

class DownloadRequestSerializer(serializers.Serializer):
    session_id = serializers.CharField(help_text="Processing session ID received from upload response")

class ReferenceDocumentDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceDocument
        fields = '__all__'  # or list specific fields
        read_only_fields = [
            'id', 'created_at', 'updated_at',  # typically system fields
            'file_path', 'extracted_text',  # add any fields you want to be read-only
            # add other fields as needed
        ]

class ReferenceDocumentUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceDocument
        fields = ['name', 'document_type']

class AudioFileDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioFile
        fields = [
            'id', 'original_filename', 'status', 'diarization_status', 'duration',
            'coverage', 'created_at', 'updated_at'
        ]

class AudioFileUpdateSerializer(serializers.Serializer):
    STATUS_CHOICES = ('pending', 'processing', 'processed', 'failed')
    DIARIZATION_CHOICES = ('pending', 'processing', 'completed', 'failed')

    original_filename = serializers.CharField(max_length=255, required=False, allow_blank=True)
    status = serializers.ChoiceField(choices=STATUS_CHOICES, required=False)
    diarization_status = serializers.ChoiceField(choices=DIARIZATION_CHOICES, required=False)
    summary = serializers.CharField(required=False, allow_blank=True)
    keywords_detected = serializers.JSONField(required=False)
    reference_document_id = serializers.UUIDField(required=False, allow_null=True)

    def validate_keywords_detected(self, value):
        if value in (None, ''):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                return [value]
        return value

class AudioFileFullSerializer(serializers.ModelSerializer):
    reference_document_id = serializers.SerializerMethodField()

    class Meta:
        model = AudioFile
        fields = [
            'id', 'original_filename', 'file_path', 'status', 'diarization_status',
            'duration', 'coverage', 'summary', 'keywords_detected',
            'reference_document_id', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'file_path', 'created_at', 'updated_at']

    def get_reference_document_id(self, obj):
        if obj.reference_document_id:
            return str(obj.reference_document_id)
        return None

    def to_representation(self, instance):
        data = super().to_representation(instance)
        keywords = data.get('keywords_detected')
        if isinstance(keywords, str):
            try:
                data['keywords_detected'] = json.loads(keywords)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return data

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

class SpeakerProfileDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpeakerProfile
        fields = ['id', 'name', 'embedding', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

class SpeakerProfileCreateUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpeakerProfile
        fields = ['name', 'embedding']
        extra_kwargs = {
            'name': {'required': True},
            'embedding': {'required': False},
        }

    def validate_embedding(self, value):
        if value is None:
            return []
        return value

    def create(self, validated_data):
        if 'embedding' not in validated_data:
            validated_data['embedding'] = []
        return super().create(validated_data)

    def update(self, instance, validated_data):
        if 'embedding' in validated_data and validated_data['embedding'] is None:
            validated_data['embedding'] = []
        return super().update(instance, validated_data)

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
