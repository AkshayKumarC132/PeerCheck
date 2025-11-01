from django.core.management.base import BaseCommand

from api.models import AudioFile, AuditLog
from api.new_utils import diarization_from_audio, build_speaker_summary


class Command(BaseCommand):
    help = 'Rerun diarization for AudioFiles that have not completed diarization.'

    def handle(self, *args, **options):
        audio_files = AudioFile.objects.filter(status='processed').exclude(diarization_status='completed')
        self.stdout.write(f'Found {audio_files.count()} audio files to process.')

        for audio_file in audio_files:
            try:
                audio_file.diarization_status = 'processing'
                audio_file.save(update_fields=['diarization_status'])
                AuditLog.objects.create(
                    action='diarization_start',
                    user=None,
                    object_id=str(audio_file.id),
                    object_type='AudioFile',
                    details={'source': 'management_rerun_diarization'},
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
                        'source': 'management_rerun_diarization',
                        'segments': len(diarization_segments),
                    },
                )
                self.stdout.write(self.style.SUCCESS(f'Processed {audio_file.id}'))
            except Exception as e:
                audio_file.diarization_status = 'failed'
                audio_file.save(update_fields=['diarization_status'])
                AuditLog.objects.create(
                    action='diarization_failed',
                    user=None,
                    object_id=str(audio_file.id),
                    object_type='AudioFile',
                    details={
                        'source': 'management_rerun_diarization',
                        'error': str(e),
                    },
                )
                self.stdout.write(self.style.ERROR(f'Error processing {audio_file.id}: {e}'))
