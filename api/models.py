from django.db import models
from django.contrib.auth.models import AbstractUser


class UserProfile(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('operator', 'Operator'),
        ('reviewer', 'Reviewer'),
    )
    id = models.AutoField(primary_key=True, db_column='user_id')
    name = models.CharField(max_length=100, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    theme = models.CharField(max_length=50, default="light")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='operator')

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
    transcription = models.JSONField(null=True, blank=True)
    status = models.CharField(max_length=50, default="pending")  # pending, processed
    keywords_detected = models.TextField(null=True, blank=True)
    duration = models.FloatField(null=True, blank=True)  # Duration in seconds
    sop = models.ForeignKey(SOP, on_delete=models.SET_NULL, null=True, blank=True, related_name='audio_files')
    user = models.ForeignKey('UserProfile', on_delete=models.SET_NULL, null=True, blank=True, related_name='audio_files_uploaded')  # New field
    summary = models.TextField(null=True, blank=True)  # Summary of the transcription
    def __str__(self):
        return self.file_path

# Feedback Model

class Feedback(models.Model):
    audio_file = models.ForeignKey(AudioFile, on_delete=models.CASCADE)
    feedback = models.CharField(max_length=50)  # complete, incomplete, needs review
    comments = models.TextField(null=True, blank=True)
    created_by = models.ForeignKey(UserProfile, on_delete=models.SET_NULL, null=True, related_name='feedbacks_submitted')
    created_at = models.DateTimeField(auto_now_add=True, null=True) # Add default for existing rows
    updated_at = models.DateTimeField(auto_now=True, null=True) # Add default for existing rows


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
    STATUS_CHOICES = (
        ('active', 'Active'),
        ('archived', 'Archived'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} by {self.user.username}"
    
class SessionUser(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='session_users')
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='session_participations')
    speaker_tag = models.CharField(max_length=50, null=True, blank=True)  # e.g., Speaker_1
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('session', 'user')

    def __str__(self):
        return f"{self.user.username} in {self.session.name}"
    
class FeedbackReview(models.Model):
    reviewer = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='feedback_reviews')
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='feedback_reviews')
    comments = models.TextField(null=True, blank=True)
    resolved_flag = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Review by {self.reviewer.username} for {self.session.name}"
    
class UserSettings(models.Model):
    user = models.OneToOneField(UserProfile, on_delete=models.CASCADE, related_name='settings')
    language = models.CharField(max_length=10, default='en')
    notification_prefs = models.JSONField(default=dict)  # e.g., {"email": true, "push": false}
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Settings for {self.user.username}"

class SystemSettings(models.Model):
    default_sop_version = models.CharField(max_length=50, default='1.0')
    timeout_threshold = models.IntegerField(default=300)  # seconds
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return "System Settings"

    class Meta:
        verbose_name_plural = "System Settings"

class AuditLog(models.Model):
    ACTION_CHOICES = (
        ('audio_upload', 'Audio Upload'),
        ('feedback_submit', 'Feedback Submit'),
        ('sop_create', 'SOP Create'),
        ('sop_update', 'SOP Update'),
        ('sop_delete', 'SOP Delete'), 
        ('audiofile_delete', 'AudioFile Delete'),
        ('feedback_update', 'Feedback Update'),
        ('feedback_delete', 'Feedback Delete'),
        ('review_submit', 'Review Submit'),
        ('feedbackreview_update', 'FeedbackReview Update'), 
        ('feedbackreview_delete', 'FeedbackReview Delete'), 
        ('userprofile_update', 'UserProfile Update'), # Added UserProfile Update
        ('userprofile_delete', 'UserProfile Delete'), # Added UserProfile Delete
        ('session_status_update', 'Session Status Update'),
        ('session_update', 'Session Update'), 
        ('session_delete', 'Session Delete'), 
    )
    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    user = models.ForeignKey(UserProfile, on_delete=models.SET_NULL, null=True, related_name='audit_logs')
    timestamp = models.DateTimeField(auto_now_add=True)
    session = models.ForeignKey(Session, on_delete=models.SET_NULL, null=True, blank=True, related_name='audit_logs')
    object_id = models.IntegerField()
    object_type = models.CharField(max_length=50)  # e.g., AudioFile, SOP, FeedbackReview
    details = models.JSONField(default=dict)  # Additional context

    def __str__(self):
        return f"{self.action} by {self.user.username if self.user else 'Unknown'} at {self.timestamp}"


class SpeakerProfile(models.Model):
    """Stores a speaker voice embedding and optional assigned name."""

    name = models.CharField(max_length=255, null=True, blank=True)
    embedding = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name or f"Speaker {self.id}"
