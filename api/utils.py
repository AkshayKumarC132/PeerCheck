# # Speech-to-Text (Vosk)

# import vosk
# import wave
# import os
# import soundfile as sf
# from pydub import AudioSegment
# from pydub.effects import normalize, low_pass_filter, high_pass_filter
import spacy # python -m spacy download en_core_web_sm
# from pydub.utils import make_chunks
# from concurrent.futures import ThreadPoolExecutor
# import logging
# import json
# from peercheck import settings

# # MODEL_PATH = os.path.join(settings.BASE_DIR, "vosk-model-en-us-0.22")
# MODEL_PATH = os.path.join(settings.BASE_DIR, "vosk-model-small-en-us-0.15")
# # vosk-model-small-en-us-0.15

# # Limit OpenBLAS threads
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"




# def transcribe_audio(file_path: str) -> str:

#     model = vosk.Model(MODEL_PATH)
#     wf = wave.open(file_path, "rb")
#     print("Wave File :",wf.getnchannels())
#     print("Wave File :",wf.getsampwidth())
#     print("Wave File :",wf.getframerate())

#     # Ensure the audio meets Vosk requirements
#     if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
#         raise ValueError("Audio must be mono, 16-bit PCM, and 16 kHz sample rate.")

#     # Initialize the recognizer
#     recognizer = vosk.KaldiRecognizer(model, wf.getframerate())
#     transcription = ""

#     # Process the audio file
#     while True:
#         data = wf.readframes(4000)  # Read in chunks of frames
#         if len(data) == 0:
#             break
        
#         if recognizer.AcceptWaveform(data):
#             # Append only the final results
#             result = json.loads(recognizer.Result())
#             transcription += result.get("text", "") + " "

#     # Add any remaining final result
#     final_result = json.loads(recognizer.FinalResult())
#     transcription += final_result.get("text", "")

#     return transcription.strip()

# # def resample_audio(file_path: str, target_sample_rate: int) -> str:
# #     """
# #     Resamples the audio file to the target sample rate.
    
# #     Args:
# #         file_path (str): Path to the original audio file.
# #         target_sample_rate (int): Desired sample rate in Hz.
        
# #     Returns:
# #         str: Path to the resampled audio file.
# #     """
# #     output_path = file_path.replace(".wav", f"_{target_sample_rate}Hz.wav")
    
# #     # Load the audio file
# #     audio = AudioSegment.from_file(file_path)
    
# #     # Resample to the target sample rate
# #     audio = audio.set_frame_rate(target_sample_rate)
    
# #     # Export the resampled audio
# #     audio.export(output_path, format="wav")
    
# #     return output_path

import os
import logging

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")

def detect_keywords(transcription, keywords):
    detected = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in transcription.lower():
            timestamp = transcription.lower().find(keyword_lower)
            detected.append({"word": keyword, "timestamp": timestamp})
    return detected

# # Segmentation

# def segment_transcription(transcription: str) -> dict:
#     words = transcription.split()
#     total_words = len(words)
#     segment_size = total_words // 3

#     return {
#         "segment_1": " ".join(words[:segment_size]),
#         "segment_2": " ".join(words[segment_size:2 * segment_size]),
#         "segment_3": " ".join(words[2 * segment_size:])
#     }












# def split_audio_in_memory(audio: AudioSegment, chunk_length_ms: int = 20000) -> list:
#     """
#     Splits audio into smaller chunks and keeps them in memory.
    
#     Args:
#         audio (AudioSegment): Preprocessed audio segment.
#         chunk_length_ms (int): Length of each chunk in milliseconds.
    
#     Returns:
#         list: List of AudioSegment chunks.
#     """
#     return make_chunks(audio, chunk_length_ms)

# def transcribe_chunk(chunk: AudioSegment, model_path: str) -> str:
#     temp_file = os.path.join("./uploads/", "temp_chunk.wav")
#     chunk.export(temp_file, format="wav")
#     return transcribe_audio(temp_file)

# def process_chunks_concurrently(chunks, model_path):
#     transcription = ""
#     with ThreadPoolExecutor(max_workers=4) as executor:  # Limit threads
#         results = executor.map(lambda chunk: transcribe_chunk(chunk, model_path), chunks)
#         transcription = " ".join(results)
#     return transcription


# # def preprocess_audio(file_path: str, target_sample_rate: int = 16000) -> str:
# #     """
# #     Preprocess the audio file: resample and set to mono if necessary.
# #     Converts invalid files to the required WAV format.
# #     """
# #     try:
# #         audio = AudioSegment.from_file(file_path)

# #         # Resample and set to mono if necessary
# #         if audio.frame_rate != target_sample_rate or audio.channels != 1:
# #             audio = audio.set_frame_rate(target_sample_rate).set_channels(1)

# #         # Export the preprocessed audio file
# #         output_path = file_path.replace(".wav", "_processed.wav")
# #         audio.export(output_path, format="wav")
# #         return output_path
# #     except Exception as e:    
# #         logging.error(f"Error preprocessing audio: {str(e)}")
# #         raise ValueError("Invalid audio file format or unsupported file.")

# # def preprocess_audio(file_path: str, target_sample_rate: int = 16000) -> str:
# #     """
# #     Preprocess the audio file: resample and set to mono if necessary.
# #     Converts invalid files to the required WAV format.
# #     """
# #     try:
# #         # Load the audio file
# #         audio = AudioSegment.from_file(file_path)

# #         # Resample to 16 kHz and set to mono
# #         if audio.frame_rate != target_sample_rate or audio.channels != 1:
# #             audio = audio.set_frame_rate(target_sample_rate).set_channels(1)

# #         # Export the preprocessed audio file
# #         output_path = file_path.replace(".wav", "_processed.wav")
# #         audio.export(output_path, format="wav")
# #         return output_path
# #     except Exception as e:
# #         logging.error(f"Error preprocessing audio: {str(e)}")
# #         raise ValueError("Invalid audio file format or unsupported file.")

# import requests
# from pydub import AudioSegment
# from io import BytesIO
# import logging

# def preprocess_audio(file_url: str, target_sample_rate: int = 16000) -> bytes:
#     """
#     Preprocess the audio file: resample and set to mono if necessary.
#     Processes audio in memory without saving to disk.
    
#     Args:
#         file_url (str): URL of the audio file.
#         target_sample_rate (int): Target sample rate (default: 16000 Hz).
    
#     Returns:
#         bytes: The processed audio file in WAV format as a byte stream.
#     """
#     try:
#         # Fetch the audio file from the URL
#         response = requests.get(file_url, stream=True)
#         if response.status_code != 200:
#             raise ValueError(f"Failed to download file from URL: {file_url}")
        
#         # Load the audio file into memory
#         audio_data = BytesIO(response.content)
#         audio = AudioSegment.from_file(audio_data)

#         # Resample to the target sample rate and set to mono
#         if audio.frame_rate != target_sample_rate or audio.channels != 1:
#             audio = audio.set_frame_rate(target_sample_rate).set_channels(1)

#         # Export the processed audio to a BytesIO object
#         processed_audio = BytesIO()
#         audio.export(processed_audio, format="wav")
#         processed_audio.seek(0)  # Reset pointer to the beginning of the stream

#         logging.info("Audio preprocessing completed successfully.")
#         return processed_audio.getvalue()  # Return the processed audio as bytes

#     except Exception as e:
#         logging.error(f"Error preprocessing audio: {str(e)}")
#         raise ValueError("Invalid audio file format or unsupported file.")

# def process_audio_pipeline(file_path: str, model_path: str) -> str:
#     """
#     Full pipeline to preprocess and transcribe an audio file.
#     """
#     # Step 1: Preprocess the audio
#     print("Preprocessing audio...")
#     preprocessed_file = preprocess_audio(file_path)

#     # Step 2: Transcribe the audio
#     print("Starting transcription...")
#     transcription = transcribe_audio(preprocessed_file)

#     print("Transcription completed.")
#     return transcription


import json
import logging
import wave
from io import BytesIO
import requests
from pydub import AudioSegment
import vosk

def preprocess_audio(file_url: str, target_sample_rate: int = 16000) -> BytesIO:
    """
    Preprocess the audio file: resample and set to mono if necessary.
    Processes audio in memory without saving to disk.
    
    Args:
        file_url (str): URL of the audio file.
        target_sample_rate (int): Target sample rate (default: 16000 Hz).
    
    Returns:
        BytesIO: The processed audio as a byte stream.
    """
    try:
        # Fetch the audio file from the URL
        response = requests.get(file_url, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to download file from URL: {file_url}")
        
        # Load the audio file into memory
        audio_data = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_data)

        # Resample to the target sample rate and set to mono
        if audio.frame_rate != target_sample_rate or audio.channels != 1:
            audio = audio.set_frame_rate(target_sample_rate).set_channels(1)

        # Export the processed audio to a BytesIO object
        processed_audio = BytesIO()
        audio.export(processed_audio, format="wav")
        processed_audio.seek(0)  # Reset pointer to the beginning of the stream

        logging.info("Audio preprocessing completed successfully.")
        return processed_audio

    except Exception as e:
        logging.error(f"Error preprocessing audio: {str(e)}")
        raise ValueError("Invalid audio file format or unsupported file.")


def transcribe_audio(audio_stream: BytesIO, model: vosk.Model) -> str:
    """
    Transcribe an audio stream using the Vosk model.

    Args:
        audio_stream (BytesIO): In-memory audio stream (WAV format).
        model (vosk.Model): Preloaded Vosk model for transcription.

    Returns:
        str: The transcription text.
    """
    try:
        # Open the audio stream as a wave file
        audio_stream.seek(0)  # Reset stream pointer
        wf = wave.open(audio_stream, "rb")

        # Ensure the audio meets Vosk requirements
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Audio must be mono, 16-bit PCM, and 16 kHz sample rate.")

        # Initialize the recognizer
        recognizer = vosk.KaldiRecognizer(model, wf.getframerate())
        transcription = ""

        # Process the audio file
        while True:
            data = wf.readframes(4000)  # Read in chunks of frames
            if len(data) == 0:
                break
            
            if recognizer.AcceptWaveform(data):
                # Append only the final results
                result = json.loads(recognizer.Result())
                transcription += result.get("text", "") + " "

        # Add any remaining final result
        final_result = json.loads(recognizer.FinalResult())
        transcription += final_result.get("text", "")

        return transcription.strip()

    except Exception as e:
        logging.error(f"Error transcribing audio: {str(e)}")
        raise ValueError("Transcription failed.")    


def process_audio_pipeline(file_url: str, model_path: str) -> str:
    """
    Full pipeline to preprocess and transcribe an audio file.

    Args:
        file_url (str): URL of the audio file.
        model_path (str): Path to the Vosk model directory.

    Returns:
        str: The transcription text.
    """
    try:
        # Step 1: Load the Vosk model
        logging.info("Loading Vosk model...")
        model = vosk.Model(model_path)

        # Step 2: Preprocess the audio
        logging.info("Preprocessing audio...")
        preprocessed_audio = preprocess_audio(file_url)

        # Step 3: Transcribe the audio
        logging.info("Starting transcription...")
        transcription = transcribe_audio(preprocessed_audio, model)

        logging.info("Transcription completed successfully.")
        return transcription

    except Exception as e:
        logging.error(f"Audio processing pipeline failed: {str(e)}")
        raise ValueError("Audio processing pipeline failed.")


from fuzzywuzzy import fuzz

def match_sop_steps(transcription, sop):
    """
    Match transcription against SOP steps and calculate confidence scores.
    
    Args:
        transcription (str): Transcribed audio text.
        sop (SOP): SOP instance to match against.
    
    Returns:
        list: List of dicts with step details and confidence scores.
    """
    results = []
    transcription_lower = transcription.lower()

    for step in sop.steps.all():
        expected_keywords = [kw.strip().lower() for kw in step.expected_keywords.split(',')]
        matches = []
        total_confidence = 0
        matched_count = 0

        for keyword in expected_keywords:
            if keyword in transcription_lower:
                # Exact match
                confidence = 100
                matched_count += 1
            else:
                # Fuzzy match for partial similarity
                best_score = 0
                for word in transcription_lower.split():
                    score = fuzz.ratio(word, keyword)
                    if score > best_score:
                        best_score = score
                confidence = best_score if best_score >= 80 else 0
                if confidence > 0:
                    matched_count += 1
            matches.append({"keyword": keyword, "confidence": confidence})
            total_confidence += confidence

        # Calculate average confidence for the step
        step_confidence = total_confidence / len(expected_keywords) if expected_keywords else 0
        results.append({
            "step_number": step.step_number,
            "instruction_text": step.instruction_text,
            "expected_keywords": expected_keywords,
            "matches": matches,
            "confidence_score": round(step_confidence, 2),
            "matched": matched_count > 0
        })

    return results

def transcribe_with_speaker_diarization(audio_url: str, model_path: str, speaker_model_path: str, session_id: int = None):
    from .models import SessionUser
    import tempfile
    logger.info(f"Starting transcription with diarization for URL: {audio_url}")
    
    # Download and preprocess audio
    try:
        response = requests.get(audio_url)
        if response.status_code != 200:
            logger.error(f"Failed to download file: {response.status_code}")
            raise ValueError(f"Failed to download file: {response.status_code}")
        
        audio_data = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_data)
        if audio.frame_rate != 16000 or audio.channels != 1:
            audio = audio.set_frame_rate(16000).set_channels(1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio.export(temp_audio, format="wav")
            temp_audio_path = temp_audio.name
        logger.info(f"Audio properties: sample_rate={audio.frame_rate}, channels={audio.channels}, duration={len(audio)/1000}s")
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {str(e)}")
        raise ValueError(f"Audio preprocessing failed: {str(e)}")

    try:
        model = vosk.Model(model_path)
        spk_model = vosk.SpkModel(speaker_model_path)
        with wave.open(temp_audio_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                logger.error("Invalid audio format for Vosk")
                raise ValueError("Audio must be mono, 16-bit PCM, 16 kHz")
            
            recognizer = vosk.KaldiRecognizer(model, wf.getframerate())
            recognizer.SetWords(True)
            recognizer.SetSpkModel(spk_model)
            transcription = []
            detected_speakers = set()
            
            # Load session users for mapping
            speaker_map = {}
            if session_id:
                session_users = SessionUser.objects.filter(session_id=session_id).order_by('created_at')
                for idx, su in enumerate(session_users):
                    speaker_tag = f"Speaker_{idx + 1}"
                    speaker_map[speaker_tag] = su.user.username
                    logger.info(f"Mapping {speaker_tag} to {su.user.username}")
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    speaker_id = None
                    if "spk" in result:
                        if isinstance(result["spk"], list) and result["spk"]:
                            speaker_id = result["spk"][0]
                        elif isinstance(result["spk"], (int, float)):
                            speaker_id = result["spk"]
                    speaker_tag = f"Speaker_{int(speaker_id) + 1}" if speaker_id is not None else "Unknown"
                    detected_speakers.add(speaker_tag)
                    speaker = speaker_map.get(speaker_tag, speaker_tag)
                    if "text" in result and result["text"]:
                        transcription.append({"speaker": speaker, "text": result["text"]})
            
            final_result = json.loads(recognizer.FinalResult())
            if "text" in final_result and final_result["text"]:
                speaker_id = None
                if "spk" in final_result:
                    if isinstance(final_result["spk"], list) and final_result["spk"]:
                        speaker_id = final_result["spk"][0]
                    elif isinstance(final_result["spk"], (int, float)):
                        speaker_id = final_result["spk"]
                speaker_tag = f"Speaker_{int(speaker_id) + 1}" if speaker_id is not None else "Unknown"
                detected_speakers.add(speaker_tag)
                speaker = speaker_map.get(speaker_tag, speaker_tag)
                transcription.append({"speaker": speaker, "text": final_result["text"]})
            
            logger.info(f"Detected speakers: {detected_speakers}")
            if len(detected_speakers) <= 1 and session_id:
                logger.warning("Only one speaker detected; diarization may have failed")
        return transcription
    except Exception as e:
        logger.error(f"Transcription with diarization failed: {str(e)}")
        raise ValueError(f"Transcription failed: {str(e)}")
    finally:
        os.remove(temp_audio_path)