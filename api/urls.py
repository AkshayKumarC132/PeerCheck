from django.urls import path
from .views import (ProcessAudioView, FeedbackView, FeedbackListView, FeedbackDetailView, 
                    FeedbackReviewListView, FeedbackReviewDetailView, 
                    GetAudioRecordsView, AudioFileDetailView, ReAnalyzeAudioView,
                    SOPCreateView, SOPListView, SOPDetailView, 
                    SessionCreateView, SessionListView, SessionDetailView, 
                    SessionReviewView, SessionStatusUpdateView, 
                    AdminUserListView, AdminUserDetailView, AdminDashboardSummaryView, # Added AdminDashboardSummaryView
                    UserSettingsView, SystemSettingsView, AuditLogView,UserProfileDetailsView,
                    SpeakerProfileUpdateView)
from .authentication import RegisterView, LoginViewAPI, LogoutViewAPI

urlpatterns = [
    path('process-audio/<str:token>/', ProcessAudioView.as_view(), name='process-audio'),
    
    # Feedback URLs
    path('submit-feedback/<str:token>/', FeedbackView.as_view(), name='submit-feedback'), # Existing POST for create
    path('feedback/<str:token>/', FeedbackListView.as_view(), name='feedback-list'), # GET list
    path('feedback/<int:feedback_id>/<str:token>/', FeedbackDetailView.as_view(), name='feedback-detail'), 

    # FeedbackReview URLs
    path('feedback-reviews/<str:token>/', FeedbackReviewListView.as_view(), name='feedback-review-list'),
    path('feedback-review/<int:review_id>/<str:token>/', FeedbackReviewDetailView.as_view(), name='feedback-review-detail'),

    path('audio-records/<str:token>/', GetAudioRecordsView.as_view(), name='audio-records'),
    path('audio-file/<int:audio_id>/<str:token>/', AudioFileDetailView.as_view(), name='audio-file-detail'),
    path('reanalyze-audio/<str:token>/', ReAnalyzeAudioView.as_view(), name='reanalyze-audio'),

    path('register/', RegisterView.as_view(), name='register'),

    path('login/', LoginViewAPI.as_view(), name='login'),

    path('logout/<str:token>/', LogoutViewAPI.as_view(), name='logout'),
    path('profile/<str:token>/', UserProfileDetailsView.as_view(), name='profile'),  # Assuming this is for user profile

    path('sop/create/<str:token>/', SOPCreateView.as_view(), name='sop-create'),
    path('sop/list/<str:token>/', SOPListView.as_view(), name='sop-list'),
    path('sop/<int:sop_id>/<str:token>/', SOPDetailView.as_view(), name='sop-detail'), 

    path('sessions/create/<str:token>/', SessionCreateView.as_view(), name='session-create'),
    path('sessions/list/<str:token>/', SessionListView.as_view(), name='session-list'),
    path('session/details/<int:session_id>/<str:token>/', SessionDetailView.as_view(), name='session-detail-main'), # New general detail view
    path('session/<int:session_id>/review/<str:token>/', SessionReviewView.as_view(), name='session-review'),
    path('session/<int:session_id>/status/<str:token>/', SessionStatusUpdateView.as_view(), name='session-status-update'),

    path('settings/user/<str:token>/', UserSettingsView.as_view(), name='user-settings'),
    path('settings/system/<str:token>/', SystemSettingsView.as_view(), name='system-settings'),
    path('audit-logs/<str:token>/', AuditLogView.as_view(), name='audit-logs'),

    # Admin User Management URLs
    path('admin/users/<str:token>/', AdminUserListView.as_view(), name='admin-user-list'),
    path('admin/user/<int:user_id>/<str:token>/', AdminUserDetailView.as_view(), name='admin-user-detail'),
    path('admin/dashboard-summary/<str:token>/', AdminDashboardSummaryView.as_view(), name='admin-dashboard-summary'), # New dashboard summary URL
    path('speaker/<int:profile_id>/<str:token>/', SpeakerProfileUpdateView.as_view(), name='speaker-update'),
]