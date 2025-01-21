from django.urls import path
from .views import ProcessAudioView, FeedbackView, GetAudioRecordsView, ReAnalyzeAudioView

urlpatterns = [
    path('process-audio/', ProcessAudioView.as_view(), name='process-audio'),
    path('submit-feedback/', FeedbackView.as_view(), name='submit-feedback'),

    path('audio-records/', GetAudioRecordsView.as_view(), name='audio-records'),

    path('reanalyze-audio/', ReAnalyzeAudioView.as_view(), name='reanalyze-audio'),
]