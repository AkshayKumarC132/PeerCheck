from celery import shared_task

from api.models import AudioFile, AuditLog
from api.new_utils import diarization_from_audio, build_speaker_summary


@shared_task
def process_missing_diarizations():
    """Ensure processed audio files eventually receive diarization results."""
    audio_files = AudioFile.objects.filter(status='processed').exclude(diarization_status='completed')
    for audio_file in audio_files:
        try:
            audio_file.diarization_status = 'processing'
            audio_file.save(update_fields=['diarization_status'])
            AuditLog.objects.create(
                action='diarization_start',
                user=None,
                object_id=str(audio_file.id),
                object_type='AudioFile',
                details={'source': 'process_missing_diarizations'},
            )

            transcript = audio_file.transcription or {}
            transcript_segments = transcript.get('segments', [])
            transcript_words = transcript.get('words', [])
            diarization_segments = diarization_from_audio(
                audio_file.file_path, transcript_segments, transcript_words
            )
            audio_file.diarization = {
                'segments': diarization_segments,
                'speakers': build_speaker_summary(diarization_segments),
            }
            audio_file.diarization_status = 'completed'
            audio_file.save(update_fields=['diarization', 'diarization_status'])
            AuditLog.objects.create(
                action='diarization_complete',
                user=None,
                object_id=str(audio_file.id),
                object_type='AudioFile',
                details={
                    'source': 'process_missing_diarizations',
                    'segments': len(diarization_segments),
                },
            )
        except Exception as e:
            audio_file.diarization_status = 'failed'
            audio_file.save(update_fields=['diarization_status'])
            AuditLog.objects.create(
                action='diarization_failed',
                user=None,
                object_id=str(audio_file.id),
                object_type='AudioFile',
                details={
                    'source': 'process_missing_diarizations',
                    'error': str(e),
                },
            )
            # Keep logging to stdout for visibility in worker logs
            print(f'Error processing {audio_file.id}: {e}')
