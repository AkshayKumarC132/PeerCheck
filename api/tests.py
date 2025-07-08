from django.test import TestCase
from rest_framework.test import APIClient
from .models import UserProfile, SOP, SOPStep, AudioFile, Session
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

class SessionTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.admin = UserProfile.objects.create_user(username='admin', email='admin@example.com', password='testpass', role='admin')
        self.user = UserProfile.objects.create_user(username='testuser', email='testuser@example.com', password='testpass', role='user')
        self.auditor = UserProfile.objects.create_user(username='auditor', email='auditor@example.com', password='testpass', role='auditor')
        self.admin_token = AuthToken.objects.create(self.admin)[1]
        self.user_token = AuthToken.objects.create(self.user)[1]
        self.auditor_token = AuthToken.objects.create(self.auditor)[1]
        self.sop = SOP.objects.create(name='Test SOP', version='1.0', created_by=self.user)
        self.audio_file = AudioFile.objects.create(file_path='test.wav', transcription='test', status='processed', duration=5)
        self.session = Session.objects.create(name='Test Session', user=self.user, sop=self.sop)

    def test_create_session_user(self):
        data = {
            "name": "Training Session",
            "sop": self.sop.id,
            "audio_file_ids": [self.audio_file.id]
        }
        response = self.client.post(f'/api/sessions/create/{self.user_token}/', data, format='json')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(Session.objects.count(), 2)
        self.assertEqual(Session.objects.last().audio_files.count(), 1)
        self.assertEqual(response.json()['name'], 'Training Session')

    def test_create_session_auditor_fails(self):
        data = {
            "name": "Training Session",
            "sop": self.sop.id
        }
        response = self.client.post(f'/api/sessions/create/{self.auditor_token}/', data, format='json')
        self.assertEqual(response.status_code, 403)

    def test_list_sessions_user(self):
        response = self.client.get(f'/api/sessions/list/{self.user_token}/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()['sessions']), 1)
        self.assertEqual(response.json()['sessions'][0]['name'], 'Test Session')

    def test_list_sessions_admin(self):
        response = self.client.get(f'/api/sessions/list/{self.admin_token}/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()['sessions']), 1)

class RoleBasedAccessTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.admin = UserProfile.objects.create_user(username='admin', email='admin@example.com', password='testpass', role='admin')
        self.user = UserProfile.objects.create_user(username='testuser', email='testuser@example.com', password='testpass', role='user')
        self.auditor = UserProfile.objects.create_user(username='auditor', email='auditor@example.com', password='testpass', role='auditor')
        self.admin_token = AuthToken.objects.create(self.admin)[1]
        self.user_token = AuthToken.objects.create(self.user)[1]
        self.auditor_token = AuthToken.objects.create(self.auditor)[1]
        self.sop = SOP.objects.create(name='Test SOP', version='1.0', created_by=self.user)
        self.audio_file = SimpleUploadedFile("test.wav", b"dummy content", content_type="audio/wav")

    def test_sop_creation_admin(self):
        data = {
            "name": "Machine SOP",
            "version": "1.0",
            "steps": [
                {"step_number": 1, "instruction_text": "Test step", "expected_keywords": "test"}
            ]
        }
        response = self.client.post(f'/api/sop/create/{self.admin_token}/', data, format='json')
        self.assertEqual(response.status_code, 201)

    def test_sop_creation_auditor_fails(self):
        data = {
            "name": "Machine SOP",
            "version": "1.0",
            "steps": [
                {"step_number": 1, "instruction_text": "Test step", "expected_keywords": "test"}
            ]
        }
        response = self.client.post(f'/api/sop/create/{self.auditor_token}/', data, format='json')
        self.assertEqual(response.status_code, 403)

    def test_process_audio_user(self):
        data = {
            'file': self.audio_file,
            'sop_id': self.sop.id
        }
        with patch('api.views.upload_file_to_s3', return_value="https://s3.amazonaws.com/test.wav"):
            with patch('api.views.transcribe_with_speaker_diarization', return_value={
                "transcription": [{"speaker": "Speaker_1", "text": "test"}]
            }):
                response = self.client.post(f'/api/process-audio/{self.user_token}/', data, format='multipart')
        self.assertEqual(response.status_code, 200)

    def test_process_audio_auditor_fails(self):
        data = {
            'file': self.audio_file,
            'sop_id': self.sop.id
        }
        response = self.client.post(f'/api/process-audio/{self.auditor_token}/', data, format='multipart')
        self.assertEqual(response.status_code, 403)

    def test_get_audio_records_auditor(self):
        response = self.client.get(f'/api/audio-records/{self.auditor_token}/')        self.assertEqual(response.status_code, 200)


class ProcedureValidationTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = UserProfile.objects.create_user(username='procuser', email='proc@example.com', password='testpass')
        self.token = AuthToken.objects.create(self.user)[1]
        self.audio_file = SimpleUploadedFile('test.wav', b'dummy audio', content_type='audio/wav')
        self.procedure_file = SimpleUploadedFile('procedure.txt', b'Step 1: say hello\nStep 2: say bye', content_type='text/plain')

    @patch('api.views.upload_file_to_s3')
    @patch('api.views.transcribe_with_speaker_diarization')
    def test_validate_procedure(self, mock_transcribe, mock_upload):
        mock_upload.return_value = 'https://s3.amazonaws.com/peercheck_files/test.wav'
        mock_transcribe.return_value = [
            {'speaker': 'Speaker_1', 'text': 'hello there'}
        ]

        data = {
            'file': self.audio_file,
            'procedure_document': self.procedure_file
        }

        response = self.client.post(f'/api/validate-procedure/{self.token}/', data, format='multipart')
        self.assertEqual(response.status_code, 200)
        self.assertIn('procedure_comparison', response.json())
        results = response.json()['procedure_comparison']['results']
        self.assertEqual(len(results), 2)
        self.assertFalse(results[1]['matched'])
        self.assertIn('highlighted_document_html', response.json()['procedure_comparison'])
