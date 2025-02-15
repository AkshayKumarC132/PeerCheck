from django.urls import path
from .views import ProcessAudioView, FeedbackView, GetAudioRecordsView, ReAnalyzeAudioView
from .authentication import register, LoginViewAPI, LogoutViewAPI

urlpatterns = [
    path('process-audio/', ProcessAudioView.as_view(), name='process-audio'),
    path('submit-feedback/', FeedbackView.as_view(), name='submit-feedback'),

    path('audio-records/<token>', GetAudioRecordsView.as_view(), name='audio-records'),

    path('reanalyze-audio/', ReAnalyzeAudioView.as_view(), name='reanalyze-audio'),

    path('register/', register, name='register'),

    path('login/', LoginViewAPI.as_view(), name='login'),
    path('logout/<str:token>', LogoutViewAPI.as_view(), name='logout'),
]