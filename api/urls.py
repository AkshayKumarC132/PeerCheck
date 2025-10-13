from django.urls import path
from .views import (
    ProcessAudioView, FeedbackView, FeedbackListView, FeedbackDetailView,
    FeedbackReviewListView, FeedbackReviewDetailView,
    GetAudioRecordsView, AudioFileDetailView, ReAnalyzeAudioView,
    SOPCreateView, SOPListView, SOPDetailView,
    SessionCreateView, SessionListView, SessionDetailView,
    SessionReviewView, SessionStatusUpdateView,
    AdminUserListView, AdminUserDetailView, AdminDashboardSummaryView,  # Added AdminDashboardSummaryView
    UserSettingsView, SystemSettingsView, AuditLogView, UserProfileDetailsView,
    SpeakerProfileUpdateView, SpeakerProfileListView, GenerateSummaryFromAudioID
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
    path('documents/<str:token>/', new_enhnaced.GetUserDocumentsView.as_view(), name='user_documents'),

    # ---------------- Audio Upload & Processing ----------------
    path('audio-records/<str:token>/', GetAudioRecordsView.as_view(), name='audio-records'),
    path('upload/<str:token>/', new_enhnaced.UploadAndProcessView.as_view(), name='upload_and_process'),

    # ----------------- Download processed DOCX with highlighted text -----------------
    path('download/<str:token>/<str:session_id>/', new_enhnaced.DownloadProcessedDocumentView.as_view(), name='download_processed'),
    path('download/with-diarization/<str:token>/<str:session_id>/', new_enhnaced.DownloadProcessedDocumentWithDiarizationView.as_view(), name='download_processed_with_diarization'),

    # ---------------- Run diarization on existing audio ----------------
    path('audio/<str:token>/diarization/run/', new_enhnaced.RunDiarizationView.as_view(), name='run-diarization'),

    # ---------------- Speaker Profile Management ----------------
    path('audio/<str:token>/diarization/map/', new_enhnaced.SpeakerProfileMappingView.as_view(), name='speaker-profile-map'),

    # ---------------- Settings, audit, profiles ----------------
    path('settings/user/<str:token>/', UserSettingsView.as_view(), name='user-settings'),
    path('settings/system/<str:token>/', SystemSettingsView.as_view(), name='system-settings'),
    path('audit-logs/<str:token>/', AuditLogView.as_view(), name='audit-logs'),

    # ---------------- Admin ----------------
    path('admin/users/<str:token>/', AdminUserListView.as_view(), name='admin-user-list'),
    path('admin/user/<int:user_id>/<str:token>/', AdminUserDetailView.as_view(), name='admin-user-detail'),
    path('admin/dashboard-summary/<str:token>/', AdminDashboardSummaryView.as_view(), name='admin-dashboard-summary'),

    # # ---------------- Core audio & processing ----------------
    # path('process-audio/<str:token>/', ProcessAudioView.as_view(), name='process-audio'),

    # # ---------------- Feedback ----------------
    # path('submit-feedback/<str:token>/', FeedbackView.as_view(), name='submit-feedback'),  # POST create
    # path('feedback/<str:token>/', FeedbackListView.as_view(), name='feedback-list'),      # GET list
    # path('feedback/<int:feedback_id>/<str:token>/', FeedbackDetailView.as_view(), name='feedback-detail'),

    # # ---------------- FeedbackReview ----------------
    # path('feedback-reviews/<str:token>/', FeedbackReviewListView.as_view(), name='feedback-review-list'),
    # path('feedback-review/<int:review_id>/<str:token>/', FeedbackReviewDetailView.as_view(), name='feedback-review-detail'),

    # ---------------- Audio records ----------------
    # path('audio-records/<str:token>/', GetAudioRecordsView.as_view(), name='audio-records'),
    # path('audio-file/<int:audio_id>/<str:token>/', AudioFileDetailView.as_view(), name='audio-file-detail'),
    # path('reanalyze-audio/<str:token>/', ReAnalyzeAudioView.as_view(), name='reanalyze-audio'),

    


    # # ---------------- SOP ----------------
    # path('sop/create/<str:token>/', SOPCreateView.as_view(), name='sop-create'),
    # path('sop/list/<str:token>/', SOPListView.as_view(), name='sop-list'),
    # path('sop/<int:sop_id>/<str:token>/', SOPDetailView.as_view(), name='sop-detail'),

    # # ---------------- Sessions ----------------
    # path('sessions/create/<str:token>/', SessionCreateView.as_view(), name='session-create'),
    # path('sessions/list/<str:token>/', SessionListView.as_view(), name='session-list'),
    # path('session/details/<int:session_id>/<str:token>/', SessionDetailView.as_view(), name='session-detail-main'),
    # path('session/<int:session_id>/review/<str:token>/', SessionReviewView.as_view(), name='session-review'),
    # path('session/<int:session_id>/status/<str:token>/', SessionStatusUpdateView.as_view(), name='session-status-update'),

    

    # ---------------- Speakers ----------------
    # path('speakers/<str:token>/', SpeakerProfileListView.as_view(), name='speaker-list'),
    # path('speaker/<int:profile_id>/<str:token>/', SpeakerProfileUpdateView.as_view(), name='speaker-update'),

    # ---------------- Summaries ----------------
    # path('audio/<str:token>/<int:audio_id>/generate-summary/', GenerateSummaryFromAudioID.as_view(), name='generate-summary-from-audio'),

    # Get specific processing session details
    # path('session/<str:token>/<uuid:session_id>/', new_enhnaced.GetProcessingSessionView.as_view(), name='get_session'),

    # # Admin cleanup of expired sessions
    # path('cleanup/<str:token>/', new_enhnaced.CleanupExpiredSessionsView.as_view(), name='cleanup_sessions'),

    # =====================================================================
    # =============== RAGitify-compatible routes (proxied) ================
    # ====================================================================

    # # ---- Assistant ----
    # path('rag/assistant/<str:token>/', new_enhnaced.RAGAssistantCreateView.as_view(), name='rag-assistant-create'),
    # path('rag/assistant/<str:token>/list/', new_enhnaced.RAGAssistantListView.as_view(), name='rag-assistant-list'),
    # path('rag/assistant/<str:token>/<str:id>/', new_enhnaced.RAGAssistantDetailView.as_view(), name='rag-assistant-detail'),

    # # ---- Thread ----
    # path('rag/thread/<str:token>/', new_enhnaced.RAGThreadCreateView.as_view(), name='rag-thread-create'),
    # path('rag/thread/<str:token>/list/', new_enhnaced.RAGThreadListView.as_view(), name='rag-thread-list'),
    # path('rag/thread/<str:token>/<str:id>/', new_enhnaced.RAGThreadDetailView.as_view(), name='rag-thread-detail'),
    # path('rag/thread/<str:token>/<str:thread_id>/messages/', new_enhnaced.RAGThreadMessagesView.as_view(), name='rag-thread-messages'),

    # # ---- Message ----
    # path('rag/message/<str:token>/', new_enhnaced.RAGMessageCreateView.as_view(), name='rag-message-create'),
    # path('rag/message/<str:token>/list/', new_enhnaced.RAGMessageListView.as_view(), name='rag-message-list'),
    # path('rag/message/<str:token>/<int:id>/', new_enhnaced.RAGMessageDetailView.as_view(), name='rag-message-detail'),

    # # ---- Run ----
    # path('rag/run/<str:token>/', new_enhnaced.RAGRunCreateView.as_view(), name='rag-run-create'),
    # path('rag/run/<str:token>/list/', new_enhnaced.RAGRunListView.as_view(), name='rag-run-list'),
    # path('rag/run/<str:token>/<str:id>/', new_enhnaced.RAGRunDetailView.as_view(), name='rag-run-detail'),
    # path('rag/run/<str:token>/<str:run_id>/submit-tool-outputs/', new_enhnaced.RAGRunSubmitToolOutputsView.as_view(), name='rag-run-submit-tool-outputs'),

]