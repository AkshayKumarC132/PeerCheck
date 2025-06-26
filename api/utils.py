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

from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import tempfile
from pydub.effects import normalize
import numpy as np

def transcribe_with_speaker_diarization(
    audio_url: str, 
    model_path: str, 
    speaker_model_path: str, 
    session_id: int = None,
    min_speaker_duration: float = 2.0,
    speaker_similarity_threshold: float = 0.85
) -> Dict[str, Any]:
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
        Dict[str, Any]: Contains the processed transcription list under
        ``transcription`` and averaged embeddings per speaker under
        ``speaker_embeddings``.
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
        processed_transcription, id_mapping = _enhance_speaker_detection(
            raw_transcription,
            speaker_map,
            expected_speakers,
            min_speaker_duration,
            speaker_similarity_threshold
        )

        speaker_embeddings = _aggregate_speaker_embeddings(raw_transcription, id_mapping)
        
        # Step 7: Post-processing and quality checks
        final_transcription = _post_process_transcription(processed_transcription, audio_info)
        
        logger.info(
            f"Transcription completed successfully. Detected {len(set(t['speaker'] for t in final_transcription))} unique speakers")
        return {
            "transcription": final_transcription,
            "speaker_embeddings": speaker_embeddings,
        }
        
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
) -> Tuple[List[Dict], Dict[int, str]]:
    """Enhanced speaker detection with clustering and validation.

    Returns both the processed transcription and a mapping of raw speaker IDs to
    final speaker tags for later use.
    """
    
    if not raw_transcription:
        return [], {}
    
    # Group by raw speaker ID and analyze patterns
    speaker_groups = defaultdict(list)
    for segment in raw_transcription:
        speaker_id = segment.get("raw_speaker_id")
        if speaker_id is not None:
            speaker_groups[speaker_id].append(segment)
    
    # Filter out speakers with insufficient content
    valid_speakers = {}
    for speaker_id, segments in speaker_groups.items():
        total_duration = sum(len(seg["text"].split()) * 0.6 for seg in segments)  # Rough duration estimate
        if total_duration >= min_duration:
            valid_speakers[speaker_id] = segments
    
    logger.info(f"Detected {len(valid_speakers)} valid speakers after filtering")
    
    # Assign final speaker labels
    processed_transcription = []
    speaker_counter = 1
    speaker_id_mapping = {}
    
    for segment in raw_transcription:
        raw_speaker_id = segment.get("raw_speaker_id")
        
        if raw_speaker_id in valid_speakers:
            if raw_speaker_id not in speaker_id_mapping:
                speaker_tag = f"Speaker_{speaker_counter}"
                speaker_id_mapping[raw_speaker_id] = speaker_tag
                speaker_counter += 1
            
            final_speaker_tag = speaker_id_mapping[raw_speaker_id]
            final_speaker_name = speaker_map.get(final_speaker_tag, final_speaker_tag)
        else:
            final_speaker_name = "Unknown"
        
        processed_transcription.append({
            "speaker": final_speaker_name,
            "text": segment["text"],
            "timestamp": segment["timestamp"],
            "confidence": segment.get("speaker_confidence", 0.0),
            "word_details": segment.get("word_details", [])
        })
    
    return processed_transcription, speaker_id_mapping

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

def _aggregate_speaker_embeddings(raw_transcription: List[Dict], mapping: Dict[int, str]) -> Dict[str, List[float]]:
    """Average speaker vectors for each final speaker tag."""
    speaker_vectors: Dict[str, List[List[float]]] = defaultdict(list)
    for segment in raw_transcription:
        raw_id = segment.get("raw_speaker_id")
        vector = segment.get("speaker_vector")
        if raw_id is None or vector is None:
            continue
        tag = mapping.get(raw_id)
        if tag:
            speaker_vectors[tag].append(vector)

    aggregated = {}
    for tag, vectors in speaker_vectors.items():
        aggregated[tag] = np.mean(np.array(vectors), axis=0).tolist()
    return aggregated

def find_matching_speaker_profile(embedding: List[float], threshold: float = 0.75):
    """Return the SpeakerProfile matching the embedding if similarity exceeds threshold.

    If a profile is matched, its stored embedding is updated by averaging it with
    the provided embedding to make future matches more robust.
    """
    from .models import SpeakerProfile

    profiles = SpeakerProfile.objects.all()
    if not profiles:
        return None, 0.0

    target = np.array(embedding)
    target_norm = np.linalg.norm(target)
    best_score = 0.0
    best_profile = None
    for profile in profiles:
        stored = np.array(profile.embedding)
        score = float(np.dot(target, stored) / (target_norm * np.linalg.norm(stored)))
        if score > best_score:
            best_score = score
            best_profile = profile

    if best_profile and best_score >= threshold:
        # Update the stored embedding by averaging with the new vector
        new_emb = np.mean([np.array(best_profile.embedding), target], axis=0)
        best_profile.embedding = new_emb.tolist()
        best_profile.save(update_fields=["embedding"])
        return best_profile, best_score

    return None, best_score

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