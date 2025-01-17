from django.db import models

# Create your models here.
class AudioFile(models.Model):
    file_path = models.CharField(max_length=255)
    transcription = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=50, default="pending")  # pending, processed
    keywords_detected = models.TextField(null=True, blank=True)
    duration = models.FloatField(null=True, blank=True)  # Duration in seconds

    def __str__(self):
        return self.file_path

# Feedback Model

class Feedback(models.Model):
    audio_file = models.ForeignKey(AudioFile, on_delete=models.CASCADE)
    feedback = models.CharField(max_length=50)  # complete, incomplete, needs review
    comments = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Feedback for {self.audio_file.id}"
