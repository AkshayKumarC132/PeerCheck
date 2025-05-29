from django.urls import path
from .views import (ProcessAudioView, FeedbackView, GetAudioRecordsView, ReAnalyzeAudioView,
                    SOPCreateView, SOPListView, SessionCreateView, SessionListView, SessionReviewView,
                    SessionStatusUpdateView, UserSettingsView, SystemSettingsView, AuditLogView)
from .authentication import RegisterView, LoginViewAPI, LogoutViewAPI

urlpatterns = [
    path('process-audio/<str:token>/', ProcessAudioView.as_view(), name='process-audio'),
    path('submit-feedback/<str:token>/', FeedbackView.as_view(), name='submit-feedback'),

    path('audio-records/<str:token>/', GetAudioRecordsView.as_view(), name='audio-records'),
    path('reanalyze-audio/<str:token>/', ReAnalyzeAudioView.as_view(), name='reanalyze-audio'),

    path('register/', RegisterView.as_view(), name='register'),

    path('login/', LoginViewAPI.as_view(), name='login'),

    path('logout/<str:token>/', LogoutViewAPI.as_view(), name='logout'),

    path('sop/create/<str:token>/', SOPCreateView.as_view(), name='sop-create'),
    path('sop/list/<str:token>/', SOPListView.as_view(), name='sop-list'),

    path('sessions/create/<str:token>/', SessionCreateView.as_view(), name='session-create'),
    path('sessions/list/<str:token>/', SessionListView.as_view(), name='session-list'),
    path('session/<int:session_id>/review/<str:token>/', SessionReviewView.as_view(), name='session-review'),
    path('session/<int:session_id>/status/<str:token>/', SessionStatusUpdateView.as_view(), name='session-status-update'),

    path('settings/user/<str:token>/', UserSettingsView.as_view(), name='user-settings'),
    path('settings/system/<str:token>/', SystemSettingsView.as_view(), name='system-settings'),
    path('audit-logs/<str:token>/', AuditLogView.as_view(), name='audit-logs'),
]