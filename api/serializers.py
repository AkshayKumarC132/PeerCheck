from rest_framework import serializers
from .models import AudioFile, Feedback

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