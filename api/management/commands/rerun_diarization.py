from django.core.management.base import BaseCommand
from api.models import AudioFile
from api.new_utils import diarization_from_audio, build_speaker_summary

class Command(BaseCommand):
    help = 'Rerun diarization for AudioFiles with null diarization'

    def handle(self, *args, **options):
        audio_files = AudioFile.objects.filter(diarization__isnull=True, status='processed')
        self.stdout.write(f'Found {audio_files.count()} audio files to process.')

        for audio_file in audio_files:
            try:
                # You may need to fetch transcript_segments and transcript_words from audio_file.transcription
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
                audio_file.save()
                self.stdout.write(self.style.SUCCESS(f'Processed {audio_file.id}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing {audio_file.id}: {e}'))