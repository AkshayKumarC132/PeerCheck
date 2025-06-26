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

class AdminUserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = [
            'id', 'username', 'email', 'name', 'role', 
            'is_active', 'is_staff', 
            'theme', # Theme can be viewed by admin
            'date_joined', 'last_login' 
        ]
        read_only_fields = ['date_joined', 'last_login', 'theme'] # Admin cannot change theme directly

    def update(self, instance, validated_data):
        # Password should not be updated here by admin; use a separate mechanism if needed.
        validated_data.pop('password', None) 
        
        # Ensure role is valid if provided
        if 'role' in validated_data and validated_data['role'] not in [choice[0] for choice in UserProfile.ROLE_CHOICES]:
            raise serializers.ValidationError({'role': 'Invalid role selected.'})

        return super().update(instance, validated_data)

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
    id = serializers.IntegerField(required=False) # Allow id for updates

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

    def update(self, instance, validated_data):
        steps_data = validated_data.pop('steps', [])

        # Update SOP instance fields
        instance.name = validated_data.get('name', instance.name)
        instance.version = validated_data.get('version', instance.version)
        instance.created_by = validated_data.get('created_by', instance.created_by)
        instance.save()

        # Handle nested SOPStep updates
        existing_steps = {step.id: step for step in instance.steps.all()}
        incoming_step_ids = set()

        for step_data in steps_data:
            step_id = step_data.get('id')
            step_number = step_data.get('step_number')

            if step_id:  # If ID is provided, it's an existing step to update
                incoming_step_ids.add(step_id)
                if step_id in existing_steps:
                    step_instance = existing_steps[step_id]

                    # Only check for duplicate step_number if it's being updated
                    if step_instance.step_number != step_number:
                        # Ensure no other step has the same step_number
                        if SOPStep.objects.filter(sop=instance, step_number=step_number).exclude(id=step_instance.id).exists():
                            raise serializers.ValidationError(f"Step number {step_number} already exists for this SOP.")

                    # Update the existing step
                    step_instance.step_number = step_data.get('step_number', step_instance.step_number)
                    step_instance.instruction_text = step_data.get('instruction_text', step_instance.instruction_text)
                    step_instance.expected_keywords = step_data.get('expected_keywords', step_instance.expected_keywords)
                    step_instance.save()
                else:
                    SOPStep.objects.create(sop=instance, **step_data)  # Create if ID is new/invalid
            else:  # No ID, so create a new step
                # Check for duplicate step_number within the same SOP
                if SOPStep.objects.filter(sop=instance, step_number=step_number).exists():
                    raise serializers.ValidationError(f"Step number {step_number} already exists for this SOP.")

                SOPStep.objects.create(sop=instance, **step_data)

        # Delete steps that are in existing_steps but not in incoming_step_ids
        steps_to_delete_ids = set(existing_steps.keys()) - incoming_step_ids
        for step_id_to_delete in steps_to_delete_ids:
            SOPStep.objects.filter(id=step_id_to_delete).delete()

        return instance

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
    created_by = UserProfileSerializer(read_only=True)
    audio_file = AudioFileSerializer(read_only=True) # Make audio_file read-only for updates, set on create
    audio_file_id = serializers.PrimaryKeyRelatedField(
        queryset=AudioFile.objects.all(), source='audio_file', write_only=True
    )

    class Meta:
        model = Feedback
        fields = ['id', 'audio_file', 'audio_file_id', 'feedback', 'comments', 'created_by', 'created_at', 'updated_at']
        read_only_fields = ['created_by', 'created_at', 'updated_at']

class SessionUserIdsSerializer(serializers.Serializer):
    userIds = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        default=[]
    )

class ProcessAudioViewSerializer(serializers.Serializer):
    file = serializers.FileField(required=True)
    sop_id = serializers.IntegerField(required=False, allow_null=True)
    session_id = serializers.IntegerField(required=False, allow_null=True)
    start_prompt = serializers.CharField(required=False, allow_blank=True)
    end_prompt = serializers.CharField(required=False, allow_blank=True)
    keywords = serializers.CharField(required=False, allow_blank=True)
    session_user_ids = SessionUserIdsSerializer(required=False)

    def validate_session_user_ids(self, value):
        # If the nested object is provided as a string, attempt to parse it
        if isinstance(value, str):
            try:
                value = json.loads(value)  # Parse the stringified JSON
            except json.JSONDecodeError:
                raise serializers.ValidationError("Invalid JSON format for session_user_ids.")
        
        # If value is None or an empty dictionary, return an empty list
        if value is None:
            return []
        
        user_ids = value.get('userIds', [])
        
        if not isinstance(user_ids, list):
            raise serializers.ValidationError("userIds must be a list of integers.")
        
        # Ensure all are integers
        try:
            print(f"Extracted user IDs: {[int(uid) for uid in user_ids]}")
            return [int(uid) for uid in user_ids]
        except ValueError:
            raise serializers.ValidationError("All userIds must be integers.")

    # def validate_session_user_ids(self, value):
    #     if value in [None, '', [], {}]:
    #         return []
    #     # Handle form-data: value might be ["[2,5]"]
    #     if isinstance(value, list) and value and isinstance(value[0], str):
    #         try:
    #             parsed_value = json.loads(value[0])
    #             if not isinstance(parsed_value, list):
    #                 raise serializers.ValidationError("session_user_ids must be a list of integers.")
    #             value = [int(x) for x in parsed_value]
    #         except json.JSONDecodeError:
    #             try:
    #                 parsed_value = ast.literal_eval(value[0])
    #                 if not isinstance(parsed_value, list):
    #                     raise serializers.ValidationError("session_user_ids must be a list of integers.")
    #                 value = [int(x) for x in parsed_value]
    #             except (ValueError, SyntaxError):
    #                 raise serializers.ValidationError("Invalid session_user_ids format. Must be a list of integers.")
    #     elif isinstance(value, str):
    #         try:
    #             parsed_value = json.loads(value)
    #             if not isinstance(parsed_value, list):
    #                 raise serializers.ValidationError("session_user_ids must be a list of integers.")
    #             value = [int(x) for x in parsed_value]
    #         except json.JSONDecodeError:
    #             try:
    #                 parsed_value = ast.literal_eval(value)
    #                 if not isinstance(parsed_value, list):
    #                     raise serializers.ValidationError("session_user_ids must be a list of integers.")
    #                 value = [int(x) for x in parsed_value]
    #             except (ValueError, SyntaxError):
    #                 raise serializers.ValidationError("Invalid session_user_ids format. Must be a list of integers.")
    #     return value

class SessionSerializer(serializers.ModelSerializer):
    audio_files = AudioFileSerializer(many=True, read_only=True)
    # sop = SOPSerializer(read_only=True) # Keep original for GET display
    user = UserProfileSerializer(read_only=True)
    
    audio_file_ids = serializers.ListField(
        child=serializers.IntegerField(), write_only=True, required=False
    )
    sop_id = serializers.PrimaryKeyRelatedField(
        queryset=SOP.objects.all(), source='sop', write_only=True, required=False, allow_null=True
    )
    # To display SOP details in GET responses, but use sop_id for write operations
    sop_details = SOPSerializer(source='sop', read_only=True)


    class Meta:
        model = Session
        fields = [
            'id', 'name', 'user', 
            'sop_details', # For reading SOP
            'sop_id',      # For writing SOP
            'status', 'audio_files', 
            'audio_file_ids', 'created_at', 'updated_at'
        ]
        # 'sop' is implicitly handled by sop_details (read) and sop_id (write)

    def create(self, validated_data):
        audio_file_ids = validated_data.pop('audio_file_ids', [])
        # sop instance is already set by PrimaryKeyRelatedField if sop_id is provided
        session = Session.objects.create(**validated_data)
        if audio_file_ids:
            audio_files = AudioFile.objects.filter(id__in=audio_file_ids)
            session.audio_files.set(audio_files)
        return session

    def update(self, instance, validated_data):
        instance.name = validated_data.get('name', instance.name)
        instance.status = validated_data.get('status', instance.status)
        
        # Update SOP if sop_id is provided
        if 'sop' in validated_data: # 'sop' will be the key due to source='sop' on sop_id field
            instance.sop = validated_data.get('sop', instance.sop)

        # Update audio files if audio_file_ids is provided
        if 'audio_file_ids' in validated_data:
            audio_file_ids = validated_data.pop('audio_file_ids', [])
            if audio_file_ids is not None: # Check if it's explicitly provided (even if empty list)
                audio_files = AudioFile.objects.filter(id__in=audio_file_ids)
                instance.audio_files.set(audio_files)
        
        instance.save()
        return instance

    def validate(self, data):
        # Name is required on create, but not necessarily on partial update
        if self.instance is None and not data.get('name'): # Check if it's a create operation
            raise serializers.ValidationError({"name": "This field is required for creating a session."})
        
        # If sop_id is provided, it's already validated by PrimaryKeyRelatedField
        # If status is provided, it should be one of the valid choices (handled by model)
        return data

class ErrorResponseSerializer(serializers.Serializer):
    error = serializers.CharField(help_text="A description of the error that occurred.")
    # Example for multiple errors (e.g. validation errors)
    # errors = serializers.DictField(
    #     child=serializers.ListField(child=serializers.CharField()),
    #     required=False,
    #     help_text="A dictionary of field-specific errors."
    # )
    detail = serializers.CharField(required=False, help_text="Sometimes used by DRF for single string error messages.")


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

    def update(self, instance, validated_data):
        # Prevent reviewer and session from being changed during an update
        validated_data.pop('reviewer_id', None) # Or raise error if present
        validated_data.pop('reviewer', None)
        validated_data.pop('session', None) # Or raise error if present

        instance.comments = validated_data.get('comments', instance.comments)
        instance.resolved_flag = validated_data.get('resolved_flag', instance.resolved_flag)
        instance.save()
        return instance


class SpeakerProfileSerializer(serializers.ModelSerializer):
    embedding_count = serializers.IntegerField(source='embeddings.count', read_only=True)

    class Meta:
        model = SpeakerProfile
        fields = ['id', 'name', 'embedding_count', 'created_at', 'updated_at']


class SpeakerEmbeddingSerializer(serializers.ModelSerializer):
    speaker = SpeakerProfileSerializer(read_only=True)
    speaker_id = serializers.IntegerField(write_only=True, required=False)

    class Meta:
        model = SpeakerEmbedding
        fields = ['id', 'vector', 'speaker', 'speaker_id', 'audio_file', 'created_at']
