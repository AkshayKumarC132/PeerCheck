from django.test import TestCase
from rest_framework.test import APIClient
from .models import UserProfile, SOP, SOPStep, AudioFile
from knox.models import AuthToken
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch
import json

class SOPTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = UserProfile.objects.create_user(username='testuser', email='test@example.com', password='testpass')
        self.token = AuthToken.objects.create(self.user)[1]
        self.sop = SOP.objects.create(name='Test SOP', version='1.0', created_by=self.user)
        self.step = SOPStep.objects.create(sop=self.sop, step_number=1, instruction_text='Say hello', expected_keywords='hello,hi')
        self.audio_file = SimpleUploadedFile("test.wav", b"dummy audio content", content_type="audio/wav")

    def test_create_sop(self):
        data = {
            'name': 'New SOP',
            'version': '1.1',
            'steps': [
                {'step_number': 1, 'instruction_text': 'Say hello', 'expected_keywords': 'hello,hi'},
            ]
        }
        response = self.client.post(f'/api/sop/create/{self.token}', data, format='json')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(SOP.objects.count(), 2)

    @patch('api.views.upload_file_to_s3')  # Mock S3 upload
    @patch('api.views.transcribe_with_speaker_diarization')  # Mock transcription
    def test_sop_matching(self, mock_transcribe, mock_upload):
        # Mock responses
        mock_upload.return_value = "https://s3.amazonaws.com/peercheck_files/test.wav"
        mock_transcribe.return_value = {
            "transcription": [
                {"speaker": "Speaker_1", "text": "hello world", "confidence": 85}
            ]
        }

        data = {
            'file': self.audio_file,
            'sop_id': self.sop.id
        }
        response = self.client.post(f'/api/process-audio/{self.token}/', data, format='multipart')
        self.assertEqual(response.status_code, 200)
        self.assertIn('sop_matches', response.json())
        self.assertGreaterEqual(response.json()['sop_matches'][0]['confidence_score'], 80)

class ProcessAudioTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = UserProfile.objects.create_user(username='testuser', email='test@example.com', password='testpass')
        self.token = AuthToken.objects.create(self.user)[1]
        self.sop = SOP.objects.create(name='Test SOP', version='1.0', created_by=self.user)
        self.step = SOPStep.objects.create(sop=self.sop, step_number=1, instruction_text='Say hello', expected_keywords='hello,hi')
        self.audio_file = SimpleUploadedFile("test.wav", b"dummy audio content", content_type="audio/wav")

    @patch('api.views.upload_file_to_s3')  # Mock S3 upload
    @patch('api.views.transcribe_with_speaker_diarization')  # Mock transcription
    def test_process_audio_with_valid_token(self, mock_transcribe, mock_upload):
        # Mock responses
        mock_upload.return_value = "https://s3.amazonaws.com/peercheck_files/test.wav"
        mock_transcribe.return_value = {
            "transcription": [{"speaker": "Speaker_1", "text": "hello world"}]
        }

        data = {
            'file': self.audio_file,
            'sop_id': self.sop.id
        }
        response = self.client.post(f'/api/process-audio/{self.token}/', data, format='multipart')
        self.assertEqual(response.status_code, 200)
        self.assertIn('transcription', response.json())
        self.assertIn('status', response.json())
        self.assertEqual(response.json()['status'], 'processed')

    def test_process_audio_with_invalid_token(self):
        data = {
            'file': self.audio_file,
            'sop_id': self.sop.id
        }
        response = self.client.post('/api/process-audio/invalid_token/', data, format='multipart')
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json())
        self.assertEqual(response.json()['error'], 'Invalid Token')