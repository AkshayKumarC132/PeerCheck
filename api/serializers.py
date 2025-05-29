from rest_framework import serializers
from .models import *
import json
import ast

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['id', 'username', 'email', 'name', 'theme', 'password', 'role']
        extra_kwargs = {
            'password': {'write_only': True}
        }

    def create(self, validated_data):
        # Hash password before saving if needed
        password = validated_data.pop('password', None)
        user = UserProfile(**validated_data)
        if password:
            user.set_password(password)
        user.save()
        return user

class UserSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserSettings
        fields = ['language', 'notification_prefs', 'created_at', 'updated_at']

class SystemSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = SystemSettings
        fields = ['default_sop_version', 'timeout_threshold', 'created_at', 'updated_at']

class AuditLogSerializer(serializers.ModelSerializer):
    user = UserProfileSerializer(read_only=True)
    session = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = AuditLog
        fields = ['id', 'action', 'user', 'timestamp', 'session', 'object_id', 'object_type', 'details']

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()

class SOPStepSerializer(serializers.ModelSerializer):
    class Meta:
        model = SOPStep
        fields = ['id', 'step_number', 'instruction_text', 'expected_keywords']

class SOPSerializer(serializers.ModelSerializer):
    steps = SOPStepSerializer(many=True)

    class Meta:
        model = SOP
        fields = ['id', 'name', 'version', 'created_by', 'created_at', 'updated_at', 'steps']

    def create(self, validated_data):
        steps_data = validated_data.pop('steps', [])
        sop = SOP.objects.create(**validated_data)
        for step_data in steps_data:
            SOPStep.objects.create(sop=sop, **step_data)
        return sop

    def validate(self, data):
        if not data.get('name'):
            raise serializers.ValidationError({"name": "This field is required."})
        if not data.get('version'):
            raise serializers.ValidationError({"version": "This field is required."})
        if not data.get('steps'):
            raise serializers.ValidationError({"steps": "At least one step is required."})
        return data

class AudioFileSerializer(serializers.ModelSerializer):
    sop = SOPSerializer(read_only=True)

    class Meta:
        model = AudioFile
        fields = ['id', 'file_path', 'transcription', 'status', 'keywords_detected', 'duration', 'sop']

class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feedback
        fields = '__all__'

class ProcessAudioViewSerializer(serializers.Serializer):
    file = serializers.FileField(required=True)
    sop_id = serializers.IntegerField(required=False, allow_null=True)
    session_id = serializers.IntegerField(required=False, allow_null=True)
    start_prompt = serializers.CharField(required=False, allow_blank=True)
    end_prompt = serializers.CharField(required=False, allow_blank=True)
    keywords = serializers.CharField(required=False, allow_blank=True)
    session_user_ids = serializers.CharField(required=False, allow_blank=True)

    def validate_session_user_ids(self, value):
        print(value)
        # Handle form-data: value might be ["[2,5]"]
        if isinstance(value, list) and value and isinstance(value[0], str):
            try:
                parsed_value = json.loads(value[0])
                if not isinstance(parsed_value, list):
                    raise serializers.ValidationError("session_user_ids must be a list of integers.")
                value = [int(x) for x in parsed_value]
            except json.JSONDecodeError:
                try:
                    parsed_value = ast.literal_eval(value[0])
                    if not isinstance(parsed_value, list):
                        raise serializers.ValidationError("session_user_ids must be a list of integers.")
                    value = [int(x) for x in parsed_value]
                except (ValueError, SyntaxError):
                    raise serializers.ValidationError("Invalid session_user_ids format. Must be a list of integers.")
        elif isinstance(value, str):
            try:
                parsed_value = json.loads(value)
                if not isinstance(parsed_value, list):
                    raise serializers.ValidationError("session_user_ids must be a list of integers.")
                value = [int(x) for x in parsed_value]
            except json.JSONDecodeError:
                try:
                    parsed_value = ast.literal_eval(value)
                    if not isinstance(parsed_value, list):
                        raise serializers.ValidationError("session_user_ids must be a list of integers.")
                    value = [int(x) for x in parsed_value]
                except (ValueError, SyntaxError):
                    raise serializers.ValidationError("Invalid session_user_ids format. Must be a list of integers.")
        return value

class SessionSerializer(serializers.ModelSerializer):
    audio_files = AudioFileSerializer(many=True, read_only=True)
    sop = SOPSerializer(read_only=True)
    user = UserProfileSerializer(read_only=True)  # Add read_only serializer for user
    audio_file_ids = serializers.ListField(
        child=serializers.IntegerField(), write_only=True, required=False
    )

    class Meta:
        model = Session
        fields = ['id', 'name', 'user', 'sop', 'status', 'audio_files', 'audio_file_ids', 'created_at', 'updated_at']

    def create(self, validated_data):
        audio_file_ids = validated_data.pop('audio_file_ids', [])
        session = Session.objects.create(**validated_data)
        if audio_file_ids:
            audio_files = AudioFile.objects.filter(id__in=audio_file_ids)
            session.audio_files.set(audio_files)
        return session

    def validate(self, data):
        if not data.get('name'):
            raise serializers.ValidationError({"name": "This field is required."})
        return data

class SessionUserSerializer(serializers.ModelSerializer):
    user = UserProfileSerializer(read_only=True)
    user_id = serializers.IntegerField()

    class Meta:
        model = SessionUser
        fields = ['id', 'user', 'user_id', 'speaker_tag', 'created_at']

    def create(self, validated_data):
        user_id = validated_data.pop('user_id')
        validated_data['user'] = UserProfile.objects.get(id=user_id)
        return super().create(validated_data)

class FeedbackReviewSerializer(serializers.ModelSerializer):
    reviewer = UserProfileSerializer(read_only=True)
    reviewer_id = serializers.IntegerField(write_only=True)

    class Meta:
        model = FeedbackReview
        fields = [
            'id', 'reviewer', 'reviewer_id', 'session', 'comments',
            'resolved_flag', 'created_at', 'updated_at'
        ]

    def create(self, validated_data):
        reviewer_id = validated_data.pop('reviewer_id')
        validated_data['reviewer'] = UserProfile.objects.get(id=reviewer_id)
        return super().create(validated_data)