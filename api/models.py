from django.db import models
from django.contrib.auth.models import AbstractUser


class UserProfile(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('user', 'User'),
        ('auditor', 'Auditor'),
    )
    id = models.AutoField(primary_key=True, db_column='user_id')
    name = models.CharField(max_length=100, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    theme = models.CharField(max_length=50, default="light")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')

    class Meta:
        db_table = "user_profile"

class SOP(models.Model):
    name = models.CharField(max_length=255)
    version = models.CharField(max_length=50)
    created_by = models.ForeignKey(UserProfile, on_delete=models.SET_NULL, null=True, related_name='sops_created')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} (v{self.version})"

class SOPStep(models.Model):
    sop = models.ForeignKey(SOP, on_delete=models.CASCADE, related_name='steps')
    step_number = models.PositiveIntegerField()
    instruction_text = models.TextField()
    expected_keywords = models.TextField()  # Comma-separated keywords
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('sop', 'step_number')  # Ensure step numbers are unique per SOP

    def __str__(self):
        return f"Step {self.step_number}: {self.instruction_text[:50]}..."


class AudioFile(models.Model):
    file_path = models.CharField(max_length=255)
    transcription = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=50, default="pending")  # pending, processed
    keywords_detected = models.TextField(null=True, blank=True)
    duration = models.FloatField(null=True, blank=True)  # Duration in seconds
    sop = models.ForeignKey(SOP, on_delete=models.SET_NULL, null=True, blank=True, related_name='audio_files')

    def __str__(self):
        return self.file_path

# Feedback Model

class Feedback(models.Model):
    audio_file = models.ForeignKey(AudioFile, on_delete=models.CASCADE)
    feedback = models.CharField(max_length=50)  # complete, incomplete, needs review
    comments = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Feedback for {self.audio_file.id}"

class KnoxAuthtoken(models.Model):
    digest = models.CharField(primary_key=True, max_length=128)
    created = models.DateTimeField()
    user = models.ForeignKey(UserProfile, models.CASCADE, null=True, blank=True, db_column='user_id')
    expiry = models.DateTimeField(blank=True, null=True)
    token_key = models.CharField(max_length=8, null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'knox_authtoken'

class Session(models.Model):
    name = models.CharField(max_length=255)
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='sessions')
    sop = models.ForeignKey(SOP, on_delete=models.SET_NULL, null=True, blank=True, related_name='sessions')
    audio_files = models.ManyToManyField(AudioFile, related_name='sessions', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} by {self.user.username}"