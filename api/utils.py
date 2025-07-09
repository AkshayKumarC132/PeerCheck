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
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from .speaker_utils import assign_speaker_profiles

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

        # Step 8: Assign speaker profiles based on embeddings
        allowed_speakers = list(speaker_map.values()) if speaker_map else None
        final_transcription = assign_speaker_profiles(
            final_transcription, allowed_speakers=allowed_speakers
        )
        
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
    
    return speaker_info

def _vector_to_speaker_id(vector: List[float], threshold: float = 0.85) -> int:
    """Convert speaker vector to speaker ID using simple clustering."""
    # This is a simplified approach - in production, you might want to use
    # more sophisticated clustering algorithms like DBSCAN or K-means
    
    # For now, use a hash-based approach for consistency
    vector_hash = hash(tuple(round(v, 3) for v in vector[:10]))  # Use first 10 components
    return abs(vector_hash) % 10  # Limit to 10 possible speakers

def _enhance_speaker_detection(
    raw_transcription: List[Dict],
    speaker_map: Dict[str, str],
    expected_speakers: int,
    min_duration: float,
    similarity_threshold: float
) -> List[Dict]:
    """Enhanced speaker detection using clustering and validation."""

    if not raw_transcription:
        return []

    # Collect embeddings for clustering
    vectors = []
    idx_map = []
    for idx, segment in enumerate(raw_transcription):
        vec = segment.get("speaker_vector")
        if isinstance(vec, list) and vec:
            vectors.append(vec)
            idx_map.append(idx)

    # Assign cluster ids using clustering if embeddings are available
    if vectors:
        vectors_np = np.array(vectors, dtype=float)
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors_np = vectors_np / norms

        if expected_speakers and expected_speakers > 0:
            clustering = AgglomerativeClustering(
                n_clusters=expected_speakers,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(vectors_np)
        else:
            clustering = DBSCAN(
                eps=max(0.05, 1 - similarity_threshold),
                min_samples=2,
                metric="cosine",
            )
            labels = clustering.fit_predict(vectors_np)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters <= 1 and len(vectors_np) > 1:
                # DBSCAN collapsed everything into one cluster or noise
                # fallback to a simple Agglomerative approach to force
                # multiple speaker groups
                fallback_clusters = expected_speakers if expected_speakers else 2
                clustering = AgglomerativeClustering(
                    n_clusters=fallback_clusters,
                    metric="cosine",
                    linkage="average",
                )
                labels = clustering.fit_predict(vectors_np)
            elif expected_speakers and n_clusters > expected_speakers:
                clustering = AgglomerativeClustering(
                    n_clusters=expected_speakers,
                    metric="cosine",
                    linkage="average",
                )
                labels = clustering.fit_predict(vectors_np)

        max_label = max(labels) if len(labels) > 0 else -1
        adjusted_labels = []
        for label in labels:
            if label == -1:
                max_label += 1
                adjusted_labels.append(max_label)
            else:
                adjusted_labels.append(int(label))

        for i, label in zip(idx_map, adjusted_labels):
            raw_transcription[i]["cluster_id"] = label
    else:
        # Fallback to raw speaker IDs
        for segment in raw_transcription:
            segment["cluster_id"] = segment.get("raw_speaker_id")

    # Group segments by cluster ID
    clusters = defaultdict(list)
    for seg in raw_transcription:
        clusters[seg.get("cluster_id")].append(seg)

    # Evaluate cluster durations (approximate) but keep all clusters
    cluster_durations = {
        cid: sum(len(s["text"].split()) * 0.6 for s in segs)
        for cid, segs in clusters.items()
    }
    logger.info(
        f"Detected {len(clusters)} speaker clusters (min duration {min_duration}s)"
    )

    # Assign final speaker labels consistently
    processed_transcription = []
    cluster_to_tag = {}
    speaker_counter = 1

    for segment in raw_transcription:
        cid = segment.get("cluster_id")

        if cid not in cluster_to_tag:
            tag = f"Speaker_{speaker_counter}"
            cluster_to_tag[cid] = tag
            speaker_counter += 1
        final_tag = cluster_to_tag[cid]
        speaker_name = speaker_map.get(final_tag, final_tag)

        processed_transcription.append({
            "speaker": speaker_name,
            "text": segment["text"],
            "timestamp": segment["timestamp"],
            "confidence": segment.get("speaker_confidence", 0.0),
            "word_details": segment.get("word_details", []),
            "speaker_vector": segment.get("speaker_vector")
        })

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
    current_vector = None
    
    for segment in filtered_transcription:
        speaker = segment["speaker"]
        text = segment["text"].strip()
        
        if speaker == current_speaker and text:
            # Merge with previous segment
            current_text += " " + text
            current_confidence = max(current_confidence, segment.get("confidence", 0))
            if not current_vector:
                current_vector = segment.get("speaker_vector")
        else:
            # Save previous segment if it exists
            if current_text:
                merged_transcription.append({
                    "speaker": current_speaker,
                    "text": current_text,
                    "timestamp": current_timestamp,
                    "confidence": current_confidence,
                    "speaker_vector": current_vector
                })
            
            # Start new segment
            current_speaker = speaker
            current_text = text
            current_timestamp = segment["timestamp"]
            current_confidence = segment.get("confidence", 0)
            current_vector = segment.get("speaker_vector")
    
    # Don't forget the last segment
    if current_text:
        merged_transcription.append({
            "speaker": current_speaker,
            "text": current_text,
            "timestamp": current_timestamp,
            "confidence": current_confidence,
            "speaker_vector": current_vector
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

import requests

def generate_summary_from_transcription(transcription, api_key=None):
    """
    Generate a summary of the transcription using an AI model (OpenAI API or HuggingFace).
    If api_key is provided, use OpenAI; otherwise, fallback to a simple extractive summary.
    """
    if not transcription:
        return ""
    # If you have an OpenAI API key, use GPT-3.5/4 for summarization
    if api_key:
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gpt-4o",  # Correct model name
                "messages": [
                    {"role": "system", "content": "Create a concise list of bullet points summarizing the key topics and important details discussed in the following meeting transcription."},
                    {"role": "user", "content": str(transcription)}
                ],
                "max_tokens": 2048,
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error using OpenAI API for summarization: {str(e)}")
            pass  # Fallback to extractive summary

    # Fallback to extractive summary if API key is not provided or request fails
    # Simple extractive summary: return the first 3 sentences or 500 characters
    import re
    sentences = re.split(r'(?<=[.!?]) +', transcription)
    summary = ' '.join(sentences[:3])
    if not summary:
        summary = transcription[:500] + ('...' if len(transcription) > 500 else '')
    return summary

def extract_text_from_document(file_obj) -> str:
    """Extract text from an uploaded procedure document.

    Supports PDF, DOCX and plain text files. Falls back to reading the file as
    UTF-8 text if specific extraction fails.
    """
    if not file_obj:
        return ""

    name = getattr(file_obj, "name", "").lower()
    extension = os.path.splitext(name)[1]
    try:
        if extension == ".pdf":
            try:
                from pdfminer.high_level import extract_text
                file_obj.seek(0)
                pdf_source = getattr(file_obj, "file", file_obj)
                pdf_path = getattr(pdf_source, "name", None)
                if pdf_path and isinstance(pdf_path, str):
                    return extract_text(pdf_path)
                else:
                    return extract_text(pdf_source)
            except Exception:
                logger.exception("PDF extraction failed")
        elif extension in [".docx"]:
            try:
                import docx
                file_obj.seek(0)
                docx_source = getattr(file_obj, "file", file_obj)
                # Always use the file-like object for docx extraction
                document = docx.Document(docx_source)
                return "\n".join(p.text for p in document.paragraphs)
            except Exception:
                logger.exception("DOCX extraction failed")

        file_obj.seek(0)
        return file_obj.read().decode("utf-8", errors="ignore")
    finally:
        file_obj.seek(0)


def compare_procedure_with_transcription(procedure_text: str, transcription_text: str) -> Dict[str, any]:
    """Compare procedure instructions with transcription text.

    Returns a dictionary containing step-by-step match details, the procedure
    document with missed steps highlighted in markdown bold and HTML in red,
    and a summary of the results.
    """
    if not procedure_text:
        return {
            "results": [],
            "highlighted_document_markdown": "",
            "highlighted_document_html": "",
            "summary": "No procedure provided."
        }

    instructions = [line.strip() for line in procedure_text.splitlines() if line.strip()]
    transcript_lower = transcription_text.lower() if transcription_text else ""
    highlighted_lines_markdown: List[str] = []
    highlighted_lines_html: List[str] = []
    results = []

    for idx, line in enumerate(instructions, start=1):
        line_lower = line.lower()
        exact = line_lower in transcript_lower
        similarity = fuzz.partial_ratio(line_lower, transcript_lower) if transcript_lower else 0
        matched = exact or similarity >= 80
        results.append({
            "step_number": idx,
            "instruction": line,
            "matched": matched,
            "similarity": similarity
        })
        if matched:
            highlighted_lines_markdown.append(line)
            highlighted_lines_html.append(line)
        else:
            highlighted_lines_markdown.append(f"**{line}**")
            highlighted_lines_html.append(f"<span style='color:red'>{line}</span>")

    missing = [r for r in results if not r["matched"]]
    summary = f"{len(missing)} of {len(results)} instructions were not mentioned in the conversation."

    return {
        "results": results,
        "highlighted_document_markdown": "\n".join(highlighted_lines_markdown),
        "highlighted_document_html": "<br>".join(highlighted_lines_html),
        "summary": summary,
    }


def find_procedure_start_index(procedure_text: str, transcription_text: str, threshold: int = 70) -> int:
    """Determine which procedure step most closely matches the start of the transcription.

    This helps align the comparison when the audio begins partway through the procedure.

    Args:
        procedure_text: Full text of the procedure document.
        transcription_text: Transcribed speech from the uploaded audio.
        threshold: Minimum fuzzy match score to consider a step as the starting point.

    Returns:
        The index (0-based) of the step that best matches the transcription start.
        If no step meets the threshold, ``0`` is returned.
    """
    instructions = [line.strip() for line in procedure_text.splitlines() if line.strip()]
    if not instructions or not transcription_text:
        return 0

    transcript_lower = transcription_text.lower()
    best_idx = 0
    best_score = 0

    for idx, line in enumerate(instructions):
        score = fuzz.partial_ratio(line.lower(), transcript_lower)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_score < threshold:
        return 0
    return best_idx


def compare_procedure_from_index(
    procedure_text: str,
    transcription_text: str,
    start_index: int = 0,
) -> Dict[str, any]:
    """Compare procedure instructions with transcription starting from ``start_index``.

    Steps before ``start_index`` are treated as missed. The rest are matched
    against the transcription text similarly to :func:`compare_procedure_with_transcription`.
    """
    if not procedure_text:
        return {
            "results": [],
            "highlighted_document_markdown": "",
            "highlighted_document_html": "",
            "summary": "No procedure provided.",
        }

    instructions = [line.strip() for line in procedure_text.splitlines() if line.strip()]
    transcript_lower = transcription_text.lower() if transcription_text else ""

    highlighted_lines_markdown: List[str] = []
    highlighted_lines_html: List[str] = []
    results = []

    for idx, line in enumerate(instructions, start=1):
        line_lower = line.lower()
        if idx - 1 >= start_index and transcript_lower:
            exact = line_lower in transcript_lower
            similarity = fuzz.partial_ratio(line_lower, transcript_lower)
            matched = exact or similarity >= 80
        else:
            similarity = 0
            matched = False

        results.append(
            {
                "step_number": idx,
                "instruction": line,
                "matched": matched,
                "similarity": similarity,
            }
        )

        if matched:
            highlighted_lines_markdown.append(line)
            highlighted_lines_html.append(line)
        else:
            highlighted_lines_markdown.append(f"**{line}**")
            highlighted_lines_html.append(f"<span style='color:red'>{line}</span>")

    missing = [r for r in results if not r["matched"]]
    summary = f"{len(missing)} of {len(results)} instructions were not mentioned in the conversation."

    return {
        "results": results,
        "highlighted_document_markdown": "\n".join(highlighted_lines_markdown),
        "highlighted_document_html": "<br>".join(highlighted_lines_html),
        "summary": summary,
    }
