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

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import tempfile
from pydub.effects import normalize

def transcribe_with_speaker_diarization(
    audio_url: str, 
    model_path: str, 
    speaker_model_path: str, 
    session_id: int = None,
    min_speaker_duration: float = 2.0,
    speaker_similarity_threshold: float = 0.85
) -> List[Dict]:
    """
    Enhanced transcription with speaker diarization supporting multiple audio formats.
    
    Args:
        audio_url (str): URL of the audio file
        model_path (str): Path to the Vosk ASR model
        speaker_model_path (str): Path to the Vosk speaker diarization model
        session_id (int, optional): Session ID for mapping speakers to users
        min_speaker_duration (float): Minimum duration (seconds) for speaker segments
        speaker_similarity_threshold (float): Threshold for speaker clustering
    
    Returns:
        List[Dict]: Transcription with speaker labels and metadata
    """
    from .models import SessionUser
    
    logger.info(f"Starting enhanced transcription with diarization for URL: {audio_url}")
    
    # Step 1: Download and validate audio
    audio_data, original_format = _download_and_validate_audio(audio_url)
    
    # Step 2: Enhanced audio preprocessing
    preprocessed_audio, audio_info = _enhanced_audio_preprocessing(audio_data, original_format)
    
    # Step 3: Create temporary file for Vosk processing
    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            preprocessed_audio.export(temp_audio, format="wav")
            temp_audio_path = temp_audio.name
        
        logger.info(f"Audio preprocessing completed. Properties: {audio_info}")
        
        # Step 4: Load session users for speaker mapping
        speaker_map = {}
        expected_speakers = 0
        if session_id:
            speaker_map, expected_speakers = _load_speaker_mapping(session_id)
        
        # Step 5: Perform transcription with speaker diarization
        raw_transcription = _perform_transcription_with_diarization(
            temp_audio_path, model_path, speaker_model_path, audio_info
        )
        
        # Step 6: Enhanced speaker processing and clustering
        processed_transcription = _enhance_speaker_detection(
            raw_transcription, 
            speaker_map, 
            expected_speakers,
            min_speaker_duration,
            speaker_similarity_threshold
        )
        
        # Step 7: Post-processing and quality checks
        final_transcription = _post_process_transcription(processed_transcription, audio_info)
        
        logger.info(f"Transcription completed successfully. Detected {len(set(t['speaker'] for t in final_transcription))} unique speakers")
        return final_transcription
        
    except Exception as e:
        logger.error(f"Transcription with diarization failed: {str(e)}")
        raise ValueError(f"Transcription failed: {str(e)}")
    finally:
        # Cleanup temporary files
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def _download_and_validate_audio(audio_url: str) -> Tuple[BytesIO, str]:
    """Download and validate audio file from URL."""
    try:
        response = requests.get(audio_url, stream=True, timeout=30)
        if response.status_code != 200:
            raise ValueError(f"Failed to download file: HTTP {response.status_code}")
        
        # Detect original format from content-type or URL extension
        content_type = response.headers.get('content-type', '').lower()
        original_format = _detect_audio_format(audio_url, content_type)
        
        audio_data = BytesIO(response.content)
        logger.info(f"Downloaded audio file: {len(response.content)} bytes, format: {original_format}")
        
        return audio_data, original_format
        
    except requests.RequestException as e:
        logger.error(f"Network error downloading audio: {str(e)}")
        raise ValueError(f"Failed to download audio file: {str(e)}")

def _detect_audio_format(url: str, content_type: str) -> str:
    """Detect audio format from URL extension or content type."""
    # Common audio formats mapping
    format_mapping = {
        'audio/wav': 'wav',
        'audio/wave': 'wav',
        'audio/mpeg': 'mp3',
        'audio/mp3': 'mp3',
        'audio/mp4': 'mp4',
        'audio/m4a': 'm4a',
        'audio/aac': 'aac',
        'audio/ogg': 'ogg',
        'audio/flac': 'flac',
        'audio/webm': 'webm'
    }
    
    # Try content type first
    if content_type in format_mapping:
        return format_mapping[content_type]
    
    # Fall back to URL extension
    extension = url.lower().split('.')[-1] if '.' in url else 'unknown'
    return extension if extension in ['wav', 'mp3', 'mp4', 'm4a', 'aac', 'ogg', 'flac', 'webm'] else 'unknown'

def _enhanced_audio_preprocessing(audio_data: BytesIO, original_format: str) -> Tuple[AudioSegment, Dict]:
    """Enhanced audio preprocessing with format detection and optimization."""
    try:
        # Load audio with format detection
        audio_data.seek(0)
        
        if original_format == 'unknown':
            # Try to load without specifying format (pydub auto-detection)
            audio = AudioSegment.from_file(audio_data)
        else:
            # Load with specific format
            audio = AudioSegment.from_file(audio_data, format=original_format)
        
        # Collect original audio information
        audio_info = {
            'original_format': original_format,
            'original_sample_rate': audio.frame_rate,
            'original_channels': audio.channels,
            'original_duration': len(audio) / 1000.0,  # Convert to seconds
            'original_bitrate': getattr(audio, 'bitrate', 'unknown'),
            'sample_width': audio.sample_width
        }
        
        logger.info(f"Original audio info: {audio_info}")
        
        # Audio quality enhancements
        processed_audio = audio
        
        # Normalize audio levels
        processed_audio = normalize(processed_audio)
        
        # Convert to mono if stereo/multi-channel
        if processed_audio.channels != 1:
            processed_audio = processed_audio.set_channels(1)
            logger.info("Converted audio to mono")
        
        # Resample to 16kHz for Vosk compatibility
        if processed_audio.frame_rate != 16000:
            processed_audio = processed_audio.set_frame_rate(16000)
            logger.info(f"Resampled audio from {audio_info['original_sample_rate']}Hz to 16000Hz")
        
        # Ensure 16-bit sample width
        if processed_audio.sample_width != 2:
            processed_audio = processed_audio.set_sample_width(2)
            logger.info("Converted to 16-bit sample width")
        
        # Audio quality filters for better speech recognition
        # Apply gentle noise reduction by filtering extreme frequencies
        if len(processed_audio) > 0:
            # High-pass filter to remove low-frequency noise
            processed_audio = processed_audio.high_pass_filter(80)
            # Low-pass filter to remove high-frequency noise
            processed_audio = processed_audio.low_pass_filter(8000)
        
        # Update audio info with processed values
        audio_info.update({
            'processed_sample_rate': processed_audio.frame_rate,
            'processed_channels': processed_audio.channels,
            'processed_duration': len(processed_audio) / 1000.0,
            'processed_sample_width': processed_audio.sample_width
        })
        
        return processed_audio, audio_info
        
    except Exception as e:
        logger.error(f"Error in audio preprocessing: {str(e)}")
        raise ValueError(f"Audio preprocessing failed: {str(e)}")

def _load_speaker_mapping(session_id: int) -> Tuple[Dict[str, str], int]:
    """Load speaker mapping from session users."""
    from .models import SessionUser
    
    speaker_map = {}
    try:
        session_users = SessionUser.objects.filter(session_id=session_id).order_by('created_at')
        for idx, su in enumerate(session_users):
            speaker_tag = f"Speaker_{idx + 1}"
            speaker_map[speaker_tag] = su.user.username
            logger.info(f"Mapping {speaker_tag} to {su.user.username}")
        
        return speaker_map, len(session_users)
    except Exception as e:
        logger.error(f"Error loading speaker mapping: {str(e)}")
        return {}, 0

def _perform_transcription_with_diarization(
    audio_path: str, 
    model_path: str, 
    speaker_model_path: str, 
    audio_info: Dict
) -> List[Dict]:
    """Perform the core transcription with speaker diarization."""
    try:
        # Load models
        model = vosk.Model(model_path)
        spk_model = vosk.SpkModel(speaker_model_path)
        
        transcription = []
        
        with wave.open(audio_path, "rb") as wf:
            # Verify audio format
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                raise ValueError("Audio format validation failed for Vosk processing")
            
            # Initialize recognizer with enhanced settings
            recognizer = vosk.KaldiRecognizer(model, wf.getframerate())
            recognizer.SetWords(True)
            recognizer.SetSpkModel(spk_model)
            
            # Process audio in chunks
            chunk_size = 4000
            frame_count = 0
            
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                
                frame_count += chunk_size
                timestamp = frame_count / wf.getframerate()
                
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result.get("text"):
                        speaker_info = _extract_speaker_info(result, timestamp)
                        transcription.append({
                            "text": result["text"],
                            "timestamp": timestamp,
                            "raw_speaker_id": speaker_info["speaker_id"],
                            "speaker_confidence": speaker_info["confidence"],
                            "speaker_vector": speaker_info.get("vector"),
                            "word_details": result.get("result", [])
                        })
            
            # Process final result
            final_result = json.loads(recognizer.FinalResult())
            if final_result.get("text"):
                speaker_info = _extract_speaker_info(final_result, timestamp)
                transcription.append({
                    "text": final_result["text"],
                    "timestamp": timestamp,
                    "raw_speaker_id": speaker_info["speaker_id"],
                    "speaker_confidence": speaker_info["confidence"],
                    "speaker_vector": speaker_info.get("vector"),
                    "word_details": final_result.get("result", [])
                })
        
        return transcription
        
    except Exception as e:
        logger.error(f"Error in core transcription: {str(e)}")
        raise

def _extract_speaker_info(result: Dict, timestamp: float) -> Dict:
    """Extract and validate speaker information from Vosk result."""
    speaker_info = {
        "speaker_id": None,
        "confidence": 0.0,
        "vector": None
    }
    
    if "spk" in result:
        spk_data = result["spk"]
        
        if isinstance(spk_data, list) and spk_data:
            # Speaker vector format
            speaker_info["vector"] = spk_data
            # Generate speaker ID from vector (simplified clustering)
            speaker_info["speaker_id"] = _vector_to_speaker_id(spk_data)
            speaker_info["confidence"] = 0.8  # Default confidence for vector-based
            
        elif isinstance(spk_data, (int, float)):
            # Direct speaker ID format
            speaker_info["speaker_id"] = int(spk_data)
            speaker_info["confidence"] = 0.9  # Higher confidence for direct ID
    
    # Handle confidence if provided separately
    if "spk_conf" in result:
        speaker_info["confidence"] = float(result["spk_conf"])
    
    # Ensure a vector is present, even if it's a placeholder or derived
    # If spk is an ID, we might not have a direct vector from Vosk here.
    # This is a simplification; ideally, the system always works with vectors for identification.
    if speaker_info["vector"] is None and speaker_info["speaker_id"] is not None:
        # Placeholder: In a real scenario, if we get an ID but no vector,
        # we'd need a way to fetch/generate a representative vector for that ID
        # or the speaker identification step would need to handle ID-based lookups too.
        # For now, we'll create a dummy vector based on the ID for consistency in structure.
        # This part highlights a potential area for deeper integration with voice embedding generation.
        speaker_info["vector"] = [float(speaker_info["speaker_id"])] * 10 # Dummy vector

    return speaker_info

def _vector_to_speaker_id(vector: List[float], threshold: float = 0.85) -> int:
    """Convert speaker vector to speaker ID using simple clustering."""
    # This is a simplified approach - in production, you might want to use
    # more sophisticated clustering algorithms like DBSCAN or K-means
    
    # For now, use a hash-based approach for consistency
    vector_hash = hash(tuple(round(v, 3) for v in vector[:10]))  # Use first 10 components
    return abs(vector_hash) % 10  # Limit to 10 possible speakers

# Speaker Identification
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .models import SpeakerProfile

def get_all_speaker_profiles():
    """Retrieves all speaker profiles from the database."""
    return SpeakerProfile.objects.all()

def identify_speaker(voice_embedding: List[float], known_speakers: List[SpeakerProfile], similarity_threshold: float = 0.75) -> Optional[str]:
    """
    Identifies a speaker by comparing the voice embedding with known speakers.

    Args:
        voice_embedding (List[float]): The voice embedding of the current speaker.
        known_speakers (List[SpeakerProfile]): A list of SpeakerProfile objects.
        similarity_threshold (float): Minimum similarity score to consider a match.

    Returns:
        Optional[str]: The name of the identified speaker, or None if no match.
    """
    if not voice_embedding or not known_speakers:
        return None

    current_embedding = np.array(voice_embedding).reshape(1, -1)

    best_match_name = None
    highest_similarity = 0.0

    for speaker_profile in known_speakers:
        profile_embedding = np.array(speaker_profile.voice_embedding).reshape(1, -1)

        # Ensure embeddings are compatible for cosine similarity (same number of dimensions)
        if current_embedding.shape[1] != profile_embedding.shape[1]:
            logger.warning(f"Skipping speaker {speaker_profile.name} due to incompatible embedding dimensions. "
                           f"Current: {current_embedding.shape[1]}, Profile: {profile_embedding.shape[1]}")
            continue

        similarity = cosine_similarity(current_embedding, profile_embedding)[0][0]

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_name = speaker_profile.name

    if highest_similarity >= similarity_threshold:
        logger.info(f"Identified speaker: {best_match_name} with similarity: {highest_similarity:.2f}")
        return best_match_name
    else:
        logger.info(f"No speaker identified above threshold {similarity_threshold}. Highest similarity: {highest_similarity:.2f} for potential match {best_match_name if best_match_name else 'None'}")
        return None


def _enhance_speaker_detection(
    raw_transcription: List[Dict],
    speaker_map: Dict[str, str], 
    expected_speakers: int,
    min_duration: float,
    similarity_threshold: float
) -> List[Dict]:
    """Enhanced speaker detection with clustering and validation."""
    
    if not raw_transcription:
        return []
    
    if not raw_transcription:
        return []

    known_speaker_profiles = get_all_speaker_profiles()
    
    # Store embeddings for each raw speaker ID
    speaker_embeddings_map = defaultdict(list)
    for segment in raw_transcription:
        raw_speaker_id = segment.get("raw_speaker_id")
        embedding = segment.get("speaker_vector")
        if raw_speaker_id is not None and embedding:
            speaker_embeddings_map[raw_speaker_id].append(embedding)

    # Determine representative embedding for each raw speaker ID
    representative_embeddings = {}
    for raw_id, embeddings_list in speaker_embeddings_map.items():
        if embeddings_list:
            # Simple average for now, more sophisticated methods could be used
            representative_embeddings[raw_id] = np.mean(embeddings_list, axis=0).tolist()

    processed_transcription = []
    unidentified_speaker_counter = 1
    # Map raw Vosk speaker IDs to final names (either identified or generic)
    final_speaker_name_map = {}

    for segment in raw_transcription:
        raw_speaker_id = segment.get("raw_speaker_id")
        voice_embedding = segment.get("speaker_vector") # Use per-segment embedding for identification attempt
        final_speaker_name = "Unknown" # Default

        if raw_speaker_id is not None:
            if raw_speaker_id in final_speaker_name_map:
                final_speaker_name = final_speaker_name_map[raw_speaker_id]
            else:
                # Attempt to identify this speaker if we haven't processed this raw_speaker_id yet
                # Use the representative embedding for this raw_speaker_id for more stable identification
                representative_embedding_for_id = representative_embeddings.get(raw_speaker_id)

                if representative_embedding_for_id:
                    identified_name = identify_speaker(representative_embedding_for_id, known_speaker_profiles, similarity_threshold)
                    if identified_name:
                        final_speaker_name = identified_name
                        logger.info(f"RawSpeakerID {raw_speaker_id} identified as: {final_speaker_name}")
                    else:
                        # Not identified in DB, assign a generic "Speaker X" label for this session
                        # Check if session_id and speaker_map can provide a name (e.g. from UI input)
                        # This part might need more sophisticated handling if multiple unidentified speakers exist
                        session_assigned_tag = f"Speaker_{unidentified_speaker_counter}" # Placeholder, may map to speaker_map

                        # Try to use speaker_map if available from session users
                        # This mapping is based on order of users in session, not voice.
                        # The true "Speaker_X" from voice might not align with "Speaker_X" from session user order.
                        # This logic needs careful review. For now, prioritize voice ID if available.

                        # If speaker_map provides a name for this generic tag, use it.
                        # Otherwise, use the generic tag itself.
                        final_speaker_name = speaker_map.get(session_assigned_tag, session_assigned_tag)

                        # Store this assignment to ensure raw_speaker_id consistently maps to this name for this transcription
                        final_speaker_name_map[raw_speaker_id] = final_speaker_name
                        logger.info(f"RawSpeakerID {raw_speaker_id} assigned generic name: {final_speaker_name}")
                        unidentified_speaker_counter += 1 # Increment for the next unidentified speaker
                else:
                    # No representative embedding, fall back to unknown or generic
                    final_speaker_name = f"Speaker_{unidentified_speaker_counter}"
                    final_speaker_name_map[raw_speaker_id] = final_speaker_name
                    logger.info(f"RawSpeakerID {raw_speaker_id} (no representative embedding) assigned generic name: {final_speaker_name}")
                    unidentified_speaker_counter += 1
        
        processed_transcription.append({
            "speaker": final_speaker_name,
            "text": segment["text"],
            "timestamp": segment["timestamp"],
            "confidence": segment.get("speaker_confidence", 0.0), # This is Vosk's confidence, not identification confidence
            "word_details": segment.get("word_details", []),
            "voice_embedding": voice_embedding # Keep the original segment embedding for potential future use (e.g. UI selection for naming)
        })

    logger.info(f"Enhance speaker detection complete. Final mapping: {final_speaker_name_map}")
    return processed_transcription

def _post_process_transcription(transcription: List[Dict], audio_info: Dict) -> List[Dict]:
    """Post-process transcription for quality and consistency."""
    
    if not transcription:
        logger.warning("Empty transcription received for post-processing")
        return []
    
    # Remove empty or very short segments
    filtered_transcription = [
        segment for segment in transcription 
        if segment.get("text", "").strip() and len(segment["text"].split()) >= 2
    ]
    
    # Merge consecutive segments from the same speaker
    merged_transcription = []
    current_speaker = None
    current_text = ""
    current_timestamp = 0
    current_confidence = 0
    
    for segment in filtered_transcription:
        speaker = segment["speaker"]
        text = segment["text"].strip()
        
        if speaker == current_speaker and text:
            # Merge with previous segment
            current_text += " " + text
            current_confidence = max(current_confidence, segment.get("confidence", 0))
        else:
            # Save previous segment if it exists
            if current_text:
                merged_transcription.append({
                    "speaker": current_speaker,
                    "text": current_text,
                    "timestamp": current_timestamp,
                    "confidence": current_confidence
                })
            
            # Start new segment
            current_speaker = speaker
            current_text = text
            current_timestamp = segment["timestamp"]
            current_confidence = segment.get("confidence", 0)
    
    # Don't forget the last segment
    if current_text:
        merged_transcription.append({
            "speaker": current_speaker,
            "text": current_text,
            "timestamp": current_timestamp,
            "confidence": current_confidence
        })
    
    # Add metadata
    unique_speakers = set(seg["speaker"] for seg in merged_transcription)
    total_duration = audio_info.get("processed_duration", 0)
    
    logger.info(f"Post-processing completed: {len(merged_transcription)} segments, "
                f"{len(unique_speakers)} unique speakers, {total_duration:.1f}s total duration")
    
    return merged_transcription

# Additional utility functions for audio format support

def validate_audio_file(file_path: str) -> Dict[str, any]:
    """Validate and get information about an audio file."""
    try:
        audio = AudioSegment.from_file(file_path)
        return {
            "valid": True,
            "duration": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "sample_width": audio.sample_width,
            "format": "detected"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

def get_supported_audio_formats() -> List[str]:
    """Get list of supported audio formats."""
    return [
        "wav", "mp3", "mp4", "m4a", "aac", 
        "ogg", "flac", "webm", "wma", "3gp"
    ]