from celery import shared_task
from api.models import AudioFile
from api.new_utils import diarization_from_audio, build_speaker_summary

@shared_task
def process_missing_diarizations():
    audio_files = AudioFile.objects.filter(diarization__isnull=True, status='processed')
    for audio_file in audio_files:
        try:
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
        except Exception as e:
            # Optionally log the error
            print(f'Error processing {audio_file.id}: {e}')