from django.db import models
from django.contrib.auth.models import AbstractUser


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

class UserProfile(AbstractUser):
    id = models.AutoField(primary_key=True, db_column='user_id')
    name = models.CharField(max_length=100, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    theme = models.CharField(max_length=50, default="light")

    class Meta:
        db_table = "user_profile"

class KnoxAuthtoken(models.Model):
    digest = models.CharField(primary_key=True, max_length=128)
    created = models.DateTimeField()
    user = models.ForeignKey(UserProfile, models.CASCADE, null=True, blank=True, db_column='user_id')
    expiry = models.DateTimeField(blank=True, null=True)
    token_key = models.CharField(max_length=8, null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'knox_authtoken'