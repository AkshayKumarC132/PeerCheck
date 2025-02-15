from rest_framework import serializers
from .models import *

class AudioFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioFile
        fields = '__all__'

class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feedback
        fields = '__all__'

class ProcessAudioViewSerializer(serializers.Serializer):
    file = serializers.FileField()

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['id', 'username', 'email', 'name','theme']

class LoginSerialzier(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()