from django.test import TestCase
from rest_framework.test import APIClient
from .models import UserProfile, SOP, SOPStep, AudioFile, Session
from knox.models import AuthToken
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch, MagicMock
import json
import numpy as np

# Assuming SpeakerProfile is in .models and identify_speaker, get_all_speaker_profiles are in .utils
# Adjust imports based on your actual file structure if tests are in a separate folder
from .models import SpeakerProfile
from .utils import identify_speaker, get_all_speaker_profiles


class SpeakerIdentificationTests(TestCase):
    def setUp(self):
        # Create some mock SpeakerProfile objects
        self.known_speakers_list = [
            MagicMock(spec=SpeakerProfile, name='Alice', voice_embedding=np.array([0.1, 0.2, 0.3, 0.4, 0.5]).tolist()),
            MagicMock(spec=SpeakerProfile, name='Bob', voice_embedding=np.array([0.6, 0.7, 0.8, 0.9, 1.0]).tolist()),
            MagicMock(spec=SpeakerProfile, name='Charlie', voice_embedding=np.array([0.2, 0.3, 0.4, 0.5, 0.6]).tolist()),
        ]

    @patch('api.utils.get_all_speaker_profiles')
    def test_identify_speaker_exact_match(self, mock_get_all_speaker_profiles):
        mock_get_all_speaker_profiles.return_value = self.known_speakers_list

        # Embedding known to be Alice's
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        identified_name = identify_speaker(test_embedding, self.known_speakers_list, similarity_threshold=0.95)
        self.assertEqual(identified_name, 'Alice')

    @patch('api.utils.get_all_speaker_profiles')
    def test_identify_speaker_close_match(self, mock_get_all_speaker_profiles):
        mock_get_all_speaker_profiles.return_value = self.known_speakers_list

        # Embedding very similar to Bob's
        test_embedding = [0.61, 0.71, 0.81, 0.91, 1.01]
        identified_name = identify_speaker(test_embedding, self.known_speakers_list, similarity_threshold=0.90)
        self.assertEqual(identified_name, 'Bob')

    @patch('api.utils.get_all_speaker_profiles')
    def test_identify_speaker_no_match_due_to_threshold(self, mock_get_all_speaker_profiles):
        mock_get_all_speaker_profiles.return_value = self.known_speakers_list

        # Embedding somewhat similar to Charlie's but below a high threshold
        test_embedding = [0.25, 0.35, 0.45, 0.55, 0.65]
        identified_name = identify_speaker(test_embedding, self.known_speakers_list, similarity_threshold=0.98)
        self.assertIsNone(identified_name)

    @patch('api.utils.get_all_speaker_profiles')
    def test_identify_speaker_no_match_dissimilar(self, mock_get_all_speaker_profiles):
        mock_get_all_speaker_profiles.return_value = self.known_speakers_list

        # Embedding very different from all known speakers
        test_embedding = [0.9, 0.1, 0.8, 0.2, 0.7]
        identified_name = identify_speaker(test_embedding, self.known_speakers_list, similarity_threshold=0.75)
        self.assertIsNone(identified_name)

    @patch('api.utils.get_all_speaker_profiles')
    def test_identify_speaker_empty_embedding(self, mock_get_all_speaker_profiles):
        mock_get_all_speaker_profiles.return_value = self.known_speakers_list
        identified_name = identify_speaker([], self.known_speakers_list)
        self.assertIsNone(identified_name)

    @patch('api.utils.get_all_speaker_profiles')
    def test_identify_speaker_no_known_speakers(self, mock_get_all_speaker_profiles):
        mock_get_all_speaker_profiles.return_value = []
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        identified_name = identify_speaker(test_embedding, [], similarity_threshold=0.75)
        self.assertIsNone(identified_name)

    @patch('api.utils.get_all_speaker_profiles')
    def test_identify_speaker_incompatible_dimensions(self, mock_get_all_speaker_profiles):
        # Alice has 5 dims, this new speaker has 4
        different_dim_speaker = [
            MagicMock(spec=SpeakerProfile, name='Dave', voice_embedding=np.array([0.1, 0.2, 0.3, 0.4]).tolist())
        ]
        mock_get_all_speaker_profiles.return_value = different_dim_speaker

        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] # 5 dims
        # Expect identify_speaker to handle this gracefully (e.g., log a warning and not match)
        identified_name = identify_speaker(test_embedding, different_dim_speaker, similarity_threshold=0.75)
        self.assertIsNone(identified_name)

        test_embedding_4_dims = [0.1, 0.2, 0.3, 0.4] # 4 dims matching Dave
        identified_name_dave = identify_speaker(test_embedding_4_dims, different_dim_speaker, similarity_threshold=0.95)
        self.assertEqual(identified_name_dave, 'Dave')


class AssignSpeakerNameViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = UserProfile.objects.create_user(username='testassignuser', password='password123', role='operator')
        self.token = AuthToken.objects.create(self.user)[1]

        # Sample transcription data that would be stored in AudioFile
        self.sample_transcription_data = [
            {"speaker": "Speaker_1", "text": "Hello this is speaker one.", "timestamp": 1.0, "confidence": 0.9, "voice_embedding": [0.1, 0.2, 0.3]},
            {"speaker": "Speaker_2", "text": "Hi, speaker two here.", "timestamp": 2.5, "confidence": 0.88, "voice_embedding": [0.4, 0.5, 0.6]},
            {"speaker": "Speaker_1", "text": "Speaking again.", "timestamp": 3.5, "confidence": 0.92, "voice_embedding": [0.11, 0.21, 0.31]} # Slightly different embedding
        ]
        self.audio_file = AudioFile.objects.create(
            file_path="test_audio/sample.wav",
            transcription=self.sample_transcription_data,
            status="processed",
            user=self.user
        )

    @patch('api.views.token_verification') # Mock token verification as it's external to this view's core logic for this test
    def test_assign_speaker_name_success(self, mock_token_verification):
        mock_token_verification.return_value = {'status': 200, 'user': self.user}

        payload = {
            "audio_file_id": self.audio_file.id,
            "generic_speaker_label": "Speaker_1",
            "assigned_name": "Alice"
        }
        response = self.client.post(f'/api/speaker/assign_name/{self.token}/', payload, format='json')

        self.assertEqual(response.status_code, 200)
        self.assertIn("SpeakerProfile created/updated", response.data["message"])

        # Verify SpeakerProfile creation
        alice_profile = SpeakerProfile.objects.get(name="Alice")
        self.assertIsNotNone(alice_profile)
        # Expected average embedding for Alice's segments: np.mean([[0.1,0.2,0.3], [0.11,0.21,0.31]], axis=0) -> [0.105, 0.205, 0.305]
        expected_embedding = [0.105, 0.205, 0.305]
        self.assertTrue(np.allclose(alice_profile.voice_embedding, expected_embedding, atol=1e-5))

        # Verify AudioFile transcription update
        updated_audio_file = AudioFile.objects.get(id=self.audio_file.id)
        for segment in updated_audio_file.transcription:
            if segment["voice_embedding"] == [0.1, 0.2, 0.3] or segment["voice_embedding"] == [0.11, 0.21, 0.31]:
                self.assertEqual(segment["speaker"], "Alice")
            elif segment["speaker"] == "Speaker_2": # Ensure other speakers are untouched
                self.assertEqual(segment["speaker"], "Speaker_2")

        # Check AuditLog (optional, good practice)
        # self.assertTrue(AuditLog.objects.filter(action='speaker_name_assigned', user=self.user, object_id=self.audio_file.id).exists())


    @patch('api.views.token_verification')
    def test_assign_speaker_name_label_not_found(self, mock_token_verification):
        mock_token_verification.return_value = {'status': 200, 'user': self.user}
        payload = {
            "audio_file_id": self.audio_file.id,
            "generic_speaker_label": "Speaker_99", # This label doesn't exist
            "assigned_name": "Bob"
        }
        response = self.client.post(f'/api/speaker/assign_name/{self.token}/', payload, format='json')
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found in transcription", response.data["error"])

    @patch('api.views.token_verification')
    def test_assign_speaker_name_audio_file_not_found(self, mock_token_verification):
        mock_token_verification.return_value = {'status': 200, 'user': self.user}
        payload = {
            "audio_file_id": 9999, # Non-existent ID
            "generic_speaker_label": "Speaker_1",
            "assigned_name": "Charlie"
        }
        response = self.client.post(f'/api/speaker/assign_name/{self.token}/', payload, format='json')
        self.assertEqual(response.status_code, 404)
        self.assertIn("AudioFile not found", response.data["error"])

    @patch('api.views.token_verification')
    def test_assign_speaker_name_missing_payload(self, mock_token_verification):
        mock_token_verification.return_value = {'status': 200, 'user': self.user}
        payload = {
            "audio_file_id": self.audio_file.id,
            # "generic_speaker_label": "Speaker_1", # Missing
            "assigned_name": "Dave"
        }
        response = self.client.post(f'/api/speaker/assign_name/{self.token}/', payload, format='json')
        self.assertEqual(response.status_code, 400)
        self.assertIn("are required", response.data["error"])

    # TODO: Add test for ProcessAudioView to ensure it correctly uses the new speaker identification
    # This would involve mocking transcribe_with_speaker_diarization to return specific embeddings
    # and then checking if the output transcription reflects identified speaker names or generic ones.


class ProcessAudioViewSingleSpeakerOverrideTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = UserProfile.objects.create_user(username='testprocessuser', password='password123', role='operator')
        self.token = AuthToken.objects.create(self.user)[1]
        self.audio_file_content = b"dummy audio content for testing"
        self.audio_file = SimpleUploadedFile("test_single_speaker.wav", self.audio_file_content, content_type="audio/wav")

    @patch('api.views.upload_file_to_s3')
    @patch('api.views.transcribe_with_speaker_diarization') # This is in api.views, which calls the one in api.utils
    @patch('api.views.token_verification')
    def test_process_audio_single_speaker_name_override(self, mock_token_verification, mock_transcribe, mock_upload_s3):
        mock_token_verification.return_value = {'status': 200, 'user': self.user}
        mock_upload_s3.return_value = "s3://fake_bucket/test_single_speaker.wav"

        # Mock transcribe_with_speaker_diarization to initially return multiple speakers
        mock_transcribe.return_value = [
            {"speaker": "Speaker_1", "text": "Segment one", "timestamp": 1.0, "confidence": 0.9, "voice_embedding": [0.1]*10},
            {"speaker": "Speaker_2", "text": "Segment two", "timestamp": 2.0, "confidence": 0.8, "voice_embedding": [0.2]*10},
            {"speaker": "Speaker_1", "text": "Segment three", "timestamp": 3.0, "confidence": 0.95, "voice_embedding": [0.11]*10}
        ]

        single_speaker_name = "TestSpeakerAlpha"
        data = {
            'file': self.audio_file,
            'speaker_names': single_speaker_name
        }

        response = self.client.post(f'/api/process-audio/{self.token}/', data, format='multipart')

        self.assertEqual(response.status_code, 200)
        response_data = response.json()

        # Check if transcription segments are all relabeled
        self.assertTrue(all(seg['speaker'] == single_speaker_name for seg in response_data['transcription']))

        # Check speaker_statistics
        speaker_stats = response_data['speaker_statistics']
        self.assertEqual(speaker_stats['total_speakers'], 1)
        self.assertEqual(len(speaker_stats['speakers']), 1)
        self.assertEqual(speaker_stats['speakers'][0], single_speaker_name)
        self.assertEqual(speaker_stats['total_segments'], 3) # Original number of segments


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
        response = self.client.get(f'/api/audio-records/{self.auditor_token}/')
        self.assertEqual(response.status_code, 200)