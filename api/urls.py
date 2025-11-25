from django.urls import path
from .views import (
    AdminDashboardSummaryView,
    AdminUserDetailView,
    AdminUserListView,
    AuditLogView,
    DashboardSummaryView,
    GetAudioRecordsView,
    SystemSettingsView,
    UserProfileDetailsView,
    UserSettingsView,
)
from .authentication import RegisterView, LoginViewAPI, LogoutViewAPI
from . import new_enhnaced

urlpatterns = [
    # ---------------- Auth (3PC) ----------------
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginViewAPI.as_view(), name='login'),
    path('logout/<str:token>/', LogoutViewAPI.as_view(), name='logout'),
    path('profile/<str:token>/', UserProfileDetailsView.as_view(), name='profile'),

    # -------------- Document Management ----------------
    path('documents/upload/<str:token>/', new_enhnaced.UploadReferenceDocumentView.as_view(), name='document-upload'),
    path('documents/<str:token>/<uuid:document_id>/', new_enhnaced.ReferenceDocumentDetailView.as_view(), name='document-detail'),
    path('documents/<str:token>/', new_enhnaced.GetUserDocumentsView.as_view(), name='user_documents'),

    # ---------------- Audio Upload & Processing ----------------
    path('audio-records/<str:token>/', GetAudioRecordsView.as_view(), name='audio-records'),
    path('upload/<str:token>/', new_enhnaced.UploadAndProcessView.as_view(), name='upload_and_process'),
    path('audio/<str:token>/<uuid:audio_id>/', new_enhnaced.AudioFileDetailView.as_view(), name='audio-detail'),

    # ----------------- Download processed DOCX with highlighted text -----------------
    path('download/<str:token>/<str:session_id>/', new_enhnaced.DownloadProcessedDocumentView.as_view(), name='download_processed'),
    path('download/with-diarization/<str:token>/<str:session_id>/', new_enhnaced.DownloadProcessedDocumentWithDiarizationView.as_view(), name='download_processed_with_diarization'),

    # ---------------- Run diarization on existing audio ----------------
    path('audio/<str:token>/diarization/run/', new_enhnaced.RunDiarizationView.as_view(), name='run-diarization'),

    # ---------------- Speaker Profile Management ----------------
    path('audio/<str:token>/diarization/map/', new_enhnaced.SpeakerProfileMappingView.as_view(), name='speaker-profile-map'),
    path('speaker-profiles/<str:token>/', new_enhnaced.SpeakerProfileListCreateView.as_view(), name='speaker-profile-list'),
    path('speaker-profiles/<str:token>/<int:profile_id>/', new_enhnaced.SpeakerProfileDetailView.as_view(), name='speaker-profile-detail'),

    # ---------------- Settings, audit, profiles ----------------
    path('settings/user/<str:token>/', UserSettingsView.as_view(), name='user-settings'),
    path('settings/system/<str:token>/', SystemSettingsView.as_view(), name='system-settings'),
    path('audit-logs/<str:token>/', AuditLogView.as_view(), name='audit-logs'),

    # ---------------- Admin ----------------
    path('admin/users/<str:token>/', AdminUserListView.as_view(), name='admin-user-list'),
    path('admin/user/<int:user_id>/<str:token>/', AdminUserDetailView.as_view(), name='admin-user-detail'),
    path('admin/dashboard-summary/<str:token>/', AdminDashboardSummaryView.as_view(), name='admin-dashboard-summary'),

    # --------------- User Dashboard -------------
    path('dashboard/summary/<str:token>/', DashboardSummaryView.as_view(), name='User Dashboard Summary')
]