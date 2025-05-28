from rest_framework import serializers
from .models import *

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['id', 'username', 'email', 'name','theme','password', 'role']

class LoginSerialzier(serializers.Serializer):
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
        fields = '__all__'

class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feedback
        fields = '__all__'

class ProcessAudioViewSerializer(serializers.Serializer):
    file = serializers.FileField()
    sop_id = serializers.IntegerField(required=False, allow_null=True)
    start_prompt = serializers.CharField(required=False, allow_blank=True)
    end_prompt = serializers.CharField(required=False, allow_blank=True)
    keywords = serializers.CharField(required=False, allow_blank=True)
    session_id = serializers.IntegerField(required=False, allow_null=True)

class SessionSerializer(serializers.ModelSerializer):
    audio_files = AudioFileSerializer(many=True, read_only=True)
    sop = SOPSerializer(read_only=True)
    audio_file_ids = serializers.ListField(
        child=serializers.IntegerField(), write_only=True, required=False
    )

    class Meta:
        model = Session
        fields = ['id', 'name', 'user', 'sop', 'audio_files', 'audio_file_ids', 'created_at', 'updated_at']

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