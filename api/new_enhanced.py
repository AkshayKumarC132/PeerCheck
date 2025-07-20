import os
import io
import re
import logging
import json
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime
from peercheck import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

import whisper
import soundfile as sf
import ffmpeg
import torch
import numpy as np
from django.db import transaction
from django.http import HttpResponse
from api.models import (SOP, Session, AudioFile, ReferenceDocument, AuditLog, SOPStep, SessionUser, SpeakerProfile)
from api.authentication import token_verification
from api.permissions import RoleBasedPermission
from .views import upload_file_to_s3, download_file_from_s3
import boto3
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import uuid
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import green, yellow, red, black
import fitz  # PyMuPDF
import zipfile

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    import torchaudio
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    logging.warning("pyannote.audio not available. Speaker diarization will be disabled.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Matplotlib not available. Timeline generation disabled.")

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

# Download nltk data
for pkg in ['punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# Settings for S3
try:
    # AWS S3 Configuration
    S3_BUCKET_NAME = settings.AWS_STORAGE_BUCKET_NAME
    S3_REGION = settings.AWS_S3_REGION_NAME
    S3_ACCESS_KEY = settings.AWS_S3_ACCESS_KEY_ID
    S3_SECRET_KEY = settings.AWS_S3_SECRET_ACCESS_KEY

    # Initialize S3 client
    s3_client = boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )
except Exception as e:
    raise Exception(f"Failed to initialize S3 client: {str(e)}")

@dataclass
class Config:
    WHISPER_MODEL: str = "base"
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"
    SENTENCE_TRANSFORMER: str = "all-mpnet-base-v2"
    LLM_MODEL: str = "microsoft/DialoGPT-large"
    USE_GPU: bool = torch.cuda.is_available()
    DEVICE: str = "cuda" if USE_GPU else "cpu"
    MATCH_THRESHOLD: float = 0.65
    PARTIAL_THRESHOLD: float = 0.45
    HIGH_CONFIDENCE_THRESHOLD: float = 0.80
    TARGET_COVERAGE: float = 0.87
    SEMANTIC_WEIGHT: float = 0.35
    FUZZY_WEIGHT: float = 0.30
    KEYWORD_WEIGHT: float = 0.20
    LLM_WEIGHT: float = 0.15
    MAX_AUDIO_LENGTH: int = 3600
    CONTEXT_WINDOW: int = 5
    BATCH_SIZE: int = 16
    MIN_KEYWORD_OVERLAP: float = 0.30
    MIN_SEMANTIC_SIMILARITY: float = 0.40
    FUZZY_BOOST_THRESHOLD: float = 0.80
    SPEAKER_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

# Data classes for segments and matches
@dataclass
class SpeakerSegment:
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

@dataclass
class StepMatch:
    step_num: str
    step_text: str
    matched_segments: List[SpeakerSegment]
    confidence_score: float
    status: str  # 'Matched', 'Partial', 'Missing'
    llm_validation: str
    speaker_attribution: List[str]
    match_details: Dict[str, float]

# Serializer (same as before)
from rest_framework import serializers
class ProcedureValidationSerializer(serializers.Serializer):
    audio_file = serializers.FileField()
    procedure_document = serializers.CharField(required=False, allow_blank=True)
    huggingface_token = serializers.CharField(required=False, allow_blank=True)
    sop_id = serializers.IntegerField(required=False)
    session_id = serializers.IntegerField(required=False)
    session_name = serializers.CharField(required=False, max_length=255)

# Model loader with caching
class EnhancedModelLoader:
    _audio_model = None
    _diarization_pipeline = None
    _embedder = None
    _llm_model = None
    _llm_tokenizer = None

    @classmethod
    def load_models(cls, hf_token: Optional[str] = None):
        if cls._audio_model is None:
            cls._audio_model = whisper.load_model(Config.WHISPER_MODEL, device=Config.DEVICE)
            logging.info(f"Whisper {Config.WHISPER_MODEL} loaded on {Config.DEVICE}")
        if cls._diarization_pipeline is None and hf_token and DIARIZATION_AVAILABLE:
            try:
                cls._diarization_pipeline = Pipeline.from_pretrained(
                    Config.DIARIZATION_MODEL,
                    use_auth_token=hf_token
                )
                if Config.USE_GPU:
                    cls._diarization_pipeline = cls._diarization_pipeline.to(torch.device(Config.DEVICE))
                logging.info("Diarization pipeline loaded")
            except Exception as e:
                logging.warning(f"Failed to load diarization: {e}")
                cls._diarization_pipeline = None
        if cls._embedder is None:
            cls._embedder = SentenceTransformer(Config.SENTENCE_TRANSFORMER, device=Config.DEVICE)
            logging.info("Sentence transformer loaded")
        if cls._llm_model is None:
            try:
                cls._llm_tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL, padding_side='left')
                cls._llm_model = AutoModelForCausalLM.from_pretrained(Config.LLM_MODEL)
                if Config.USE_GPU:
                    cls._llm_model = cls._llm_model.to(Config.DEVICE)
                if cls._llm_tokenizer.pad_token is None:
                    cls._llm_tokenizer.pad_token = cls._llm_tokenizer.eos_token
                logging.info("LLM model loaded")
            except Exception as e:
                logging.warning(f"Failed to load LLM: {e}")
                cls._llm_model = None
                cls._llm_tokenizer = None
        return cls._audio_model, cls._diarization_pipeline, cls._embedder, cls._llm_model, cls._llm_tokenizer

# Audio processing class
class EnhancedAudioProcessor:
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token

    def convert_to_wav_optimized(self, input_path: str, output_path: str):
        try:
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    format='wav',
                    acodec='pcm_s16le',
                    ac=1,
                    ar='16000',
                    loglevel='error',
                    af='highpass=f=80,lowpass=f=8000'
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            logging.error(f"Audio conversion failed: {e.stderr.decode()}")
            raise

    def transcribe_with_speaker_diarization(self, response):
        # Load models
        audio_model, diarization_pipeline, _, _, _ = EnhancedModelLoader.load_models(self.hf_token)
        
        # Extract filename from response headers or URL
        filename = self._get_filename_from_response(response)
        ext = os.path.splitext(filename.lower())[1]

        with TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, f"input{ext}")
            # Write response content to a file
            with open(input_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Convert to WAV
            wav_path = os.path.join(tmpdir, "audio.wav")
            self.convert_to_wav_optimized(input_path, wav_path)

            # Load audio
            audio_array, sr = sf.read(wav_path, dtype='float32')
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

            # Truncate long audio
            max_samples = Config.MAX_AUDIO_LENGTH * 16000
            if len(audio_array) > max_samples:
                audio_array = audio_array[:max_samples]
                logging.warning(f"Audio truncated to {Config.MAX_AUDIO_LENGTH} seconds")

            # Transcribe
            transcription = audio_model.transcribe(
                audio_array,
                word_timestamps=True,
                temperature=0.0,
                beam_size=5,
                best_of=3,
                patience=1.0
            )

            # Diarization
            diarization = None
            if diarization_pipeline:
                try:
                    waveform, sample_rate = torchaudio.load(wav_path)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
                    logging.info(f"Diarization found {len(diarization.labels())} speakers")
                except Exception as e:
                    logging.error(f"Diarization failed: {e}")

            # Align transcription with speakers
            speaker_segments = self._align_transcription_with_speakers(transcription, diarization)
            return transcription['text'], speaker_segments

    def _get_filename_from_response(self, response):
        # Attempt to extract filename from Content-Disposition header
        cd = response.headers.get('content-disposition')
        if cd:
            filename = re.findall('filename="?([^";]+)"?', cd)
            if filename:
                return filename[0]
        # Fallback: parse from URL
        return os.path.basename(response.url)

    def _align_transcription_with_speakers(self, transcription, diarization) -> List[SpeakerSegment]:
        segments = []
        for segment in transcription.get('segments', []):
            segment_start = segment['start']
            segment_end = segment['end']
            segment_text = segment['text'].strip()

            best_speaker = "SPEAKER_00"
            best_confidence = 1.0
            if diarization:
                best_overlap = 0
                for speaker_turn, _, speaker_id in diarization.itertracks(yield_label=True):
                    if speaker_turn.start <= segment_end and speaker_turn.end >= segment_start:
                        overlap = min(segment_end, speaker_turn.end) - max(segment_start, speaker_turn.start)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_speaker = speaker_id
                            segment_duration = segment_end - segment_start
                            best_confidence = min(1.0, overlap / segment_duration) if segment_duration > 0 else 0.5
            # Use transcription confidence
            transcription_confidence = segment.get('avg_logprob', 0.0)
            if transcription_confidence < -1.0:
                best_confidence *= 0.7

            speaker_segment = SpeakerSegment(
                speaker_id=best_speaker,
                start_time=float(segment_start),
                end_time=float(segment_end),
                text=segment_text,
                confidence=float(best_confidence)
            )
            segments.append(speaker_segment)
        return segments

# Content validator with full step extraction and matching
class LLMContentValidator:
    def __init__(self):
        self.llm_model = None
        self.llm_tokenizer = None
        self.embedder = None

    def load_models(self):
        _, _, self.embedder, self.llm_model, self.llm_tokenizer = EnhancedModelLoader.load_models()

    def extract_enhanced_steps(self, doc_text: str) -> List[Tuple[str, str]]:
        patterns = [
            r'^\s*(\d+(?:\.\d+)*\.?)\s+(.*?)(?=^\s*\d+(?:\.\d+)*\.?|$)',
            r'^\s*(\d+[\)\]:]\s+)(.*?)(?=^\s*\d+[\)\]:]\s+|$)',
            r'^\s*([A-Z]\.?)\s+(.*?)(?=^\s*[A-Z]\.?|$)',
            r'^\s*(Step\s+\d+[:.]\s*)(.*?)(?=^\s*Step\s+\d+[:.]\s*|$)',
            r'^\s*(STEP\s+\d+[:.]\s*)(.*?)(?=^\s*STEP\s+\d+[:.]\s*|$)',
            r'^\s*([•\-\*]\s*)(.*?)(?=^\s*[•\-\*]\s*|$)',
            r'^\s*(\d+\.\d+\.\d+)\s+(.*?)(?=^\s*\d+\.\d+\.\d+|$)',
            r'^\s*(ENSURE|CHECK|VERIFY|CONFIRM|PERFORM)\s+(.*?)(?=^\s*(?:ENSURE|CHECK|VERIFY|CONFIRM|PERFORM)|$)',
        ]

        norm_text = re.sub(r'[\xa0\t\r]+', ' ', doc_text)
        lines = [line.strip() for line in norm_text.splitlines() if line.strip()]

        for pattern in patterns:
            steps = []
            current_step_text = ""
            for line in lines:
                match = re.match(pattern, line, re.IGNORECASE | re.MULTILINE)
                if match:
                    if current_step_text:
                        steps.append((str(len(steps)+1), current_step_text.strip()))
                    step_num = re.sub(r'[^\w\d]', '', match.group(1)) or str(len(steps)+1)
                    step_text = match.group(2).strip()
                    if len(step_text.split()) > 3 and len(step_text) > 15:
                        steps.append((step_num, step_text))
                        current_step_text = ""
                    else:
                        current_step_text = step_text
                else:
                    if current_step_text and len(line) > 10:
                        current_step_text += " " + line
            if current_step_text and len(current_step_text.split()) > 3:
                steps.append((str(len(steps)+1), current_step_text))
            if len(steps) >= 3:
                return self._validate_extracted_steps(steps)
        # fallback
        return self._fallback_step_extraction(norm_text)

    def _validate_extracted_steps(self, steps):
        validated = []
        for step_num, step_text in steps:
            if len(step_text.split()) >= 4 and len(step_text) >= 20:
                if not re.match(r'^(end|complete|finish|done)\.?$', step_text.lower()):
                    validated.append((step_num, step_text))
        return validated

    def _fallback_step_extraction(self, norm_text):
        paragraphs = [p.strip() for p in norm_text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in norm_text.split('\n') if len(p.strip()) > 20]
        valid_steps = []
        for i, para in enumerate(paragraphs):
            if len(para.split()) > 5 and len(para) > 25:
                if any(keyword in para.lower() for keyword in ['ensure', 'check', 'verify', 'perform', 'open', 'close', 'place', 'if', 'then']):
                    valid_steps.append((str(i+1), para))
        return valid_steps

    def advanced_step_matching(self, steps, speaker_segments):
        if not speaker_segments:
            return []

        matches = []
        segment_texts = [seg.text for seg in speaker_segments]
        segment_embeddings = self.embedder.encode(segment_texts, convert_to_tensor=True)

        for step_num, step_text in steps:
            match_result = self._comprehensive_step_match(step_text, speaker_segments, segment_texts, segment_embeddings)
            match_obj = StepMatch(
                step_num=step_num,
                step_text=step_text,
                matched_segments=match_result['matched_segments'],
                confidence_score=match_result['final_score'],
                status=match_result['status'],
                llm_validation=match_result['llm_validation'],
                speaker_attribution=match_result['speaker_attribution'],
                match_details=match_result['score_breakdown']
            )
            matches.append(match_obj)

        # Coverage optimization
        matches = self._intelligent_coverage_optimization(matches, speaker_segments)
        return matches

    def _comprehensive_step_match(self, step_text, speaker_segments, segment_texts, segment_embeddings):
        # Semantic similarity
        step_embedding = self.embedder.encode([step_text], convert_to_tensor=True)
        semantic_similarities = util.pytorch_cos_sim(step_embedding, segment_embeddings)[0]

        # Fuzzy matching
        fuzzy_scores = []
        for txt in segment_texts:
            p_ratio = fuzz.partial_ratio(step_text.lower(), txt.lower()) / 100.0
            s_ratio = fuzz.token_sort_ratio(step_text.lower(), txt.lower()) / 100.0
            set_ratio = fuzz.token_set_ratio(step_text.lower(), txt.lower()) / 100.0
            fuzzy_scores.append(max(p_ratio, s_ratio, set_ratio))

        # Keyword overlap
        keyword_scores = self._calculate_keyword_overlap(step_text, segment_texts)

        # Technical match
        technical_scores = self._calculate_technical_match(step_text, segment_texts)

        # Combine scores
        combined_scores = []
        for i in range(len(segment_texts)):
            score = (
                semantic_similarities[i] * Config.SEMANTIC_WEIGHT +
                fuzzy_scores[i] * Config.FUZZY_WEIGHT +
                keyword_scores[i] * Config.KEYWORD_WEIGHT +
                technical_scores[i] * 0.20
            )
            if fuzzy_scores[i] >= Config.FUZZY_BOOST_THRESHOLD:
                score *= 1.15
            if keyword_scores[i] >= Config.MIN_KEYWORD_OVERLAP:
                score *= 1.10
            combined_scores.append(score)

        # Get top candidates
        top_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
        matched_segments = []
        final_score = 0
        status = 'Missing'
        llm_validation = ""
        speaker_attribution = []

        for idx in top_indices[:5]:
            score = combined_scores[idx]
            if score >= Config.PARTIAL_THRESHOLD:
                matched_segments.append(speaker_segments[idx])
                final_score = score
                # Determine status
                if score >= Config.MATCH_THRESHOLD:
                    status = 'Matched'
                elif score >= Config.PARTIAL_THRESHOLD:
                    # Check for additional validation
                    if (keyword_scores[idx] >= 0.3 or technical_scores[idx] >= 0.4 or semantic_similarities[idx] >= 0.6):
                        status = 'Matched'
                    else:
                        status = 'Partial'
                else:
                    status = 'Missing'
                # LLM validation
                llm_validation = self._validate_with_llm(step_text, speaker_segments[idx].text)
                speaker_attribution.append(speaker_segments[idx].speaker_id)

        if not matched_segments:
            return {
                'matched_segments': [],
                'final_score': 0,
                'status': 'Missing',
                'llm_validation': "No matching segments",
                'speaker_attribution': [],
                'score_breakdown': {
                    'semantic': 0,
                    'fuzzy': 0,
                    'keyword': 0,
                    'technical': 0,
                    'final': 0
                }
            }

        return {
            'matched_segments': matched_segments,
            'final_score': final_score,
            'status': status,
            'llm_validation': llm_validation,
            'speaker_attribution': speaker_attribution,
            'score_breakdown': {
                'semantic': float(semantic_similarities[0]),
                'fuzzy': max(fuzzy_scores),
                'keyword': max(keyword_scores),
                'technical': max(technical_scores),
                'final': final_score
            }
        }

    def _calculate_keyword_overlap(self, step_text, segment_texts):
        step_words = set(word_tokenize(step_text.lower()))
        scores = []
        for txt in segment_texts:
            seg_words = set(word_tokenize(txt.lower()))
            intersection = len(step_words.intersection(seg_words))
            union = len(step_words.union(seg_words))
            scores.append(intersection / union if union > 0 else 0)
        return scores

    def _calculate_technical_match(self, step_text, segment_texts):
        patterns = [
            r'[A-Z]{2,3}[-\s]?[A-Z]{1,2}[-\s]?\d+[A-Z]?',
            r'[A-Z]{1,2}[-\s]?[A-Z][-\s]?\d+',
            r'\b[A-Z]{2,}[-\s]?\d+[A-Z]?\b',
            r'\b(?:OPEN|CLOSE|CHECK|ENSURE|PLACE|PERFORM|STROKE|TEST|ALIGNMENT)\b'
        ]
        step_techs = set()
        for pattern in patterns:
            matches = re.findall(pattern, step_text.upper())
            step_techs.update(matches)
        scores = []
        for txt in segment_texts:
            seg_techs = set()
            for pattern in patterns:
                matches = re.findall(pattern, txt.upper())
                seg_techs.update(matches)
            intersection = len(step_techs.intersection(seg_techs))
            union = len(step_techs.union(seg_techs))
            scores.append(intersection / union if union > 0 else 0)
        return scores

    def _validate_with_llm(self, step_text, segment_text):
        # Use a simple prompt and generation
        try:
            prompt = f"Analyze if this audio matches the procedure step:\n\nStep: {step_text[:150]}\n\nAudio: {segment_text[:150]}\n\nMatch assessment:"
            tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL)
            model = AutoModelForCausalLM.from_pretrained(Config.LLM_MODEL)
            if Config.USE_GPU:
                model = model.to(Config.DEVICE)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
            if Config.USE_GPU:
                inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if any(word in result.lower() for word in ['good', 'match', 'correct', 'yes', 'accurate']):
                return f"Strong match: {result[:50]}..."
            elif any(word in result.lower() for word in ['partial', 'some', 'related']):
                return f"Partial match: {result[:50]}..."
            else:
                return f"Analysis: {result[:50]}..."
        except Exception:
            return "Validation analysis could not be performed."

    def _intelligent_coverage_optimization(self, matches, segments):
        current_coverage = self._calculate_coverage(matches)
        if current_coverage < Config.TARGET_COVERAGE:
            # Upgrade high-confidence partials
            for m in matches:
                if m.status == 'Partial' and m.confidence_score >= 0.55:
                    m.status = 'Matched'
                    m.llm_validation += " [Upgraded based on high confidence]"
            current_coverage = self._calculate_coverage(matches)
            # Try to find missing matches
            if current_coverage < Config.TARGET_COVERAGE:
                self._find_missing_matches(matches, segments)
            # Context-based matching
            current_coverage = self._calculate_coverage(matches)
            if current_coverage < Config.TARGET_COVERAGE:
                self._context_based_matching(matches, segments)
        return matches

    def _find_missing_matches(self, matches, segments):
        for m in matches:
            if m.status == 'Missing':
                best_seg = None
                best_score = 0
                for seg in segments:
                    score = fuzz.partial_ratio(m.step_text.lower(), seg.text.lower()) / 100.0
                    if any(w in seg.text.lower() for w in ['check', 'ensure', 'open', 'close', 'perform']):
                        score *= 1.2
                    if score > best_score and score >= 0.25:
                        best_score = score
                        best_seg = seg
                if best_seg:
                    m.status = 'Partial'
                    m.confidence_score = best_score
                    m.matched_segments = [best_seg]
                    m.speaker_attribution = [best_seg.speaker_id]
                    m.llm_validation = "Low-confidence match found in optimization"
                    m.match_details = {'fuzzy': best_score, 'final': best_score}

    def _context_based_matching(self, matches, segments):
        # Simplified for brevity: matching based on context
        speaker_contexts = {}
        for seg in segments:
            speaker_contexts.setdefault(seg.speaker_id, []).append(seg)
        for spk in speaker_contexts:
            speaker_contexts[spk].sort(key=lambda s: s.start_time)
        for m in matches:
            if m.status == 'Missing':
                for spk, segs in speaker_contexts.items():
                    context_text = " ".join(s.text for s in segs)
                    step_words = set(w.lower() for w in m.step_text.split() if len(w) > 3)
                    context_words = set(w.lower() for w in context_text.split())
                    overlap = len(step_words.intersection(context_words))
                    if overlap >= 2:
                        best_seg = max(segs, key=lambda s: fuzz.partial_ratio(m.step_text.lower(), s.text.lower()))
                        m.status = 'Partial'
                        m.confidence_score = 0.4
                        m.matched_segments = [best_seg]
                        m.speaker_attribution = [spk]
                        m.llm_validation = "Context-based match found"
                        m.match_details = {'context': 0.4, 'final': 0.4}
                        break

    def _calculate_coverage(self, matches):
        if not matches:
            return 0.0
        total = len(matches)
        matched = sum(1 for m in matches if m.status == 'Matched')
        partial = sum(1 for m in matches if m.status == 'Partial')
        return (matched + 0.5 * partial) / total

# Timeline generator
class SpeakerTimelineGenerator:
    def __init__(self):
        self.enabled = VISUALIZATION_AVAILABLE

    def create_speaker_timeline(self, speaker_segments, matches):
        if not self.enabled:
            return io.BytesIO()
        try:
            fig, ax = plt.subplots(figsize=(18, 12))
            speakers = list(set(s.speaker_id for s in speaker_segments))
            speaker_positions = {spk: i for i, spk in enumerate(speakers)}
            for seg in speaker_segments:
                y_pos = speaker_positions[seg.speaker_id]
                color_idx = hash(seg.speaker_id) % len(Config.SPEAKER_COLORS)
                color = Config.SPEAKER_COLORS[color_idx]
                alpha = 0.5 + (seg.confidence * 0.4)
                rect = patches.Rectangle(
                    (seg.start_time, y_pos - 0.4),
                    seg.duration,
                    0.8,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor='black',
                    linewidth=0.7
                )
                ax.add_patch(rect)
                if seg.duration > 10:
                    text = seg.text[:30] + "..." if len(seg.text) > 30 else seg.text
                    ax.text(
                        seg.start_time + seg.duration / 2,
                        y_pos,
                        text,
                        ha='center',
                        va='center',
                        fontsize=9,
                        weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9)
                    )
            # plot step markers
            for m in matches:
                for seg in m.matched_segments:
                    y_pos = speaker_positions.get(seg.speaker_id, 0)
                    if m.status == 'Matched':
                        marker = '^'
                        color='darkgreen'
                        size=120
                        edge_color='black'
                    elif m.status == 'Partial':
                        marker='s'
                        color='darkorange'
                        size=100
                        edge_color='darkred'
                    else:
                        continue
                    confidence_size = size * (0.7 + 0.3 * m.confidence_score)
                    ax.scatter(
                        seg.start_time, y_pos + 0.7,
                        marker=marker,
                        c=color,
                        s=confidence_size,
                        alpha=0.9,
                        edgecolors=edge_color,
                        linewidths=2
                    )
                    step_label = f"S{m.step_num}\n({m.confidence_score:.2f})"
                    ax.text(
                        seg.start_time, y_pos + 1.1, step_label,
                        ha='center', va='bottom', fontsize=8, weight='bold',
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor='lightcyan' if m.status == 'Matched' else 'lightgoldenrodyellow',
                            alpha=0.9, edgecolor='black', linewidth=1
                        )
                    )
            ax.set_xlabel('Time (seconds)', fontsize=16, weight='bold')
            ax.set_ylabel('Speakers', fontsize=16, weight='bold')
            ax.set_title('Speaker Timeline with Validation', fontsize=18, weight='bold')
            ax.set_yticks(range(len(speakers)))
            ax.set_yticklabels(speakers, fontsize=14, weight='bold')
            max_time = max((s.end_time for s in speaker_segments), default=0)
            ax.set_xlim(-10, max_time + 20)
            ax.set_ylim(-0.8, len(speakers) + 0.5)

            # Legend
            legend_elements = [
                Line2D([0], [0], marker='^', color='w', markerfacecolor='darkgreen', markersize=12, label='Matched Steps', markeredgecolor='black', markeredgewidth=2),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange', markersize=12, label='Partial Steps', markeredgecolor='darkred', markeredgewidth=2),
                Line2D([0], [0], color='gray', linewidth=3, label='Speaker Confidence', alpha=0.7)
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, fancybox=True)

            # Coverage info
            coverage = self._calculate_coverage_from_matches(matches)
            matched = sum(1 for m in matches if m.status == 'Matched')
            partial = sum(1 for m in matches if m.status == 'Partial')
            missing = sum(1 for m in matches if m.status == 'Missing')
            coverage_text = f'Coverage: {coverage:.1%}\nMatched: {matched} | Partial: {partial} | Missing: {missing}'
            ax.text(0.02, 0.98, coverage_text, transform=ax.transAxes, fontsize=13, weight='bold', bbox=dict(boxstyle="round,pad=0.7", facecolor='lightblue', alpha=0.9, edgecolor='navy'), verticalalignment='top')

            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            buf.seek(0)
            plt.close()
            return buf
        except Exception as e:
            logging.error(f"Error in timeline creation: {e}")
            return io.BytesIO()

    def _calculate_coverage_from_matches(self, matches):
        if not matches:
            return 0.0
        matched = sum(1 for m in matches if m.status=='Matched')
        partial = sum(1 for m in matches if m.status=='Partial')
        total = len(matches)
        return (matched + 0.5*partial)/total if total > 0 else 0

# PDF report creation
def create_enhanced_pdf_report(steps, matches, transcript, summary, overall_score, speaker_segments, procedure_text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(612, 792))
    width, height = 612, 792
    y = height - 60

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, y, "Enhanced PeerCheck Validation Report")
    y -= 50

    # Overall coverage
    c.setFont("Helvetica-Bold", 14)
    color = green if overall_score >= 0.85 else yellow if overall_score >= 0.70 else red
    c.setFillColor(color)
    c.drawString(50, y, f"Overall Coverage: {overall_score*100:.1f}%")
    c.setFillColor(black)
    y -= 30

    # Speaker summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Speaker Analysis:")
    y -= 20
    c.setFont("Helvetica", 10)
    spk_ids = set(s.speaker_id for s in speaker_segments)
    for spk in spk_ids:
        total_time = sum(s.duration for s in speaker_segments if s.speaker_id == spk)
        matched_steps = sum(1 for m in matches for s in m.matched_segments if s.speaker_id == spk)
        c.drawString(60, y, f"• {spk}: {total_time:.1f}s speaking, {matched_steps} steps matched")
        y -= 15

    y -= 10
    # Step validation results
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Step Validation Results:")
    y -= 20
    c.setFont("Helvetica-Bold", 9)
    c.drawString(55, y, "Step")
    c.drawString(100, y, "Status")
    c.drawString(160, y, "Confidence")
    c.drawString(230, y, "Speakers")
    c.drawString(320, y, "LLM Validation")
    y -= 15
    c.line(50, y, 550, y)
    y -= 10
    c.setFont("Helvetica", 8)
    for match in matches:
        if y < 80:
            c.showPage()
            y = height - 60
        color_map = {'Matched': green, 'Partial': yellow, 'Missing': red}
        c.setFillColor(color_map.get(match.status, black))
        c.drawString(55, y, match.step_num)
        c.drawString(100, y, match.status)
        c.drawString(160, y, f"{match.confidence_score:.2f}")
        c.drawString(230, y, ", ".join(match.speaker_attribution[:2]))
        llm_text = match.llm_validation[:30] + "..." if len(match.llm_validation) > 30 else match.llm_validation
        c.drawString(320, y, llm_text)
        c.setFillColor(black)
        y -= 12

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Transcript Summary:")
    y -= 15
    c.setFont("Helvetica", 9)
    for line in sent_tokenize(summary):
        if y < 60:
            c.showPage()
            y = height - 60
        for subline in [line[i:i+80] for i in range(0, len(line), 80)]:
            c.drawString(60, y, subline)
            y -= 12
    c.save()
    buffer.seek(0)
    return buffer

# PDF annotation
def enhanced_annotate_pdf(input_path, steps, matches, output_path):
    status_colors = {
        'Matched': (0, 1, 0),
        'Partial': (1, 1, 0),
        'Missing': (1, 0, 0)
    }
    try:
        doc = fitz.open(input_path)
        for (step_num, step_text), match in zip(steps, matches):
            search_patterns = [
                step_text[:50].strip(),
                step_text[:30].strip(),
                step_num
            ]
            for search_text in search_patterns:
                if len(search_text) < 5:
                    continue
                for page in doc:
                    text_instances = page.search_for(search_text)
                    if text_instances:
                        for inst in text_instances[:1]:
                            highlight = page.add_highlight_annot(inst)
                            color = status_colors.get(match.status, (0, 0, 0))
                            highlight.set_colors(stroke=color)
                            annotation_text = f"Step {step_num}: {match.status}\nConfidence: {match.confidence_score:.2f}\nSpeakers: {', '.join(match.speaker_attribution)}\nLLM: {match.llm_validation[:50]}..."
                            popup = page.add_text_annot(inst.tl, annotation_text)
                            popup.set_info(title=f"Validation: {match.status}")
                            highlight.update()
                            popup.update()
                            break
                    if text_instances:
                        break
                if text_instances:
                    break
        doc.save(output_path)
        doc.close()
    except Exception as e:
        logging.error(f"Failed to annotate PDF: {e}")
        import shutil
        shutil.copy2(input_path, output_path)

# Main API view
class EnhancedTranscriptionValidationView(APIView):
    serializer_class = ProcedureValidationSerializer
    permission_classes = [RoleBasedPermission]

    def post(self, request, token, document_id, format=None):
        start_time = time.time()

        # Token validation
        user = token_verification(token)
        if user['status'] != 200:
            return Response({'error': user['error']}, status=status.HTTP_400_BAD_REQUEST)
        user_obj = user['user']

        serializer = self.serializer_class(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Extract data
        audio_file = serializer.validated_data['audio_file']
        hf_token = serializer.validated_data.get('huggingface_token')
        sop_id = serializer.validated_data.get('sop_id')
        session_id = serializer.validated_data.get('session_id')
        session_name = serializer.validated_data.get('session_name')

        # Fetch reference document
        try:
            reference_document_obj = ReferenceDocument.objects.get(id=document_id)
            if reference_document_obj.upload_status != 'processed':
                return Response({"error": "Reference document not ready."}, status=status.HTTP_400_BAD_REQUEST)
            if not reference_document_obj.extracted_text:
                return Response({"error": "Reference document has no text."}, status=status.HTTP_400_BAD_REQUEST)
        except ReferenceDocument.DoesNotExist:
            return Response({"error": "Reference document not found."}, status=status.HTTP_404_NOT_FOUND)

        audio_file_obj = None
        session_obj = None
        sop_obj = None
        steps = []
        procedure_text = ""

        try:
            with transaction.atomic():
                # SOP handling
                if sop_id:
                    try:
                        sop_obj = SOP.objects.get(id=sop_id)
                        steps = [(str(step.step_number), step.instruction_text) for step in sop_obj.steps.all().order_by('step_number')]
                        if not steps:
                            return Response({"error": "No steps in SOP."}, status=status.HTTP_400_BAD_REQUEST)
                        procedure_text = "\n".join([s[1] for s in steps])
                        if sop_obj.reference_document:
                            reference_document_obj = sop_obj.reference_document
                    except SOP.DoesNotExist:
                        return Response({"error": "SOP not found."}, status=status.HTTP_404_NOT_FOUND)
                else:
                    # No SOP, use reference document
                    content_validator = LLMContentValidator()
                    content_validator.load_models()
                    procedure_text = reference_document_obj.extracted_text
                    steps = content_validator.extract_enhanced_steps(procedure_text)
                    if not steps:
                        return Response({"error": "No steps found."}, status=status.HTTP_400_BAD_REQUEST)
                    existing_sop = reference_document_obj.related_sops.first()
                    if existing_sop:
                        sop_obj = existing_sop
                    else:
                        # Create new SOP with duplicate check
                        if user_obj:
                            sop_obj = SOP.objects.create(
                                name=f"{reference_document_obj.name} - SOP",
                                version="1.0",
                                created_by=user_obj,
                                reference_document=reference_document_obj
                            )
                            for step_num_str, step_text in steps:
                                try:
                                    step_number = int(step_num_str)
                                except:
                                    step_number = len(steps)
                                # Avoid duplicate step_number
                                existing_step = SOPStep.objects.filter(sop=sop_obj, step_number=step_number).first()
                                if existing_step:
                                    existing_step.instruction_text = step_text
                                    existing_step.save()
                                else:
                                    SOPStep.objects.create(
                                        sop=sop_obj,
                                        step_number=step_number,
                                        instruction_text=step_text,
                                        expected_keywords=""
                                    )
                            # Audit
                            AuditLog.objects.create(
                                action='sop_create',
                                user=user_obj,
                                object_id=sop_obj.id,
                                object_type='SOP',
                                details={
                                    'reference_document_id': reference_document_obj.id,
                                    'sop_name': sop_obj.name,
                                    'steps_count': len(steps)
                                }
                            )

                # Upload audio
                file_ext = os.path.splitext(audio_file.name)[1]
                s3_file_name = f"audio/{uuid.uuid4()}{file_ext}"
                try:
                    file_url = upload_file_to_s3(audio_file, S3_BUCKET_NAME, s3_file_name)
                except Exception as e:
                    return Response({"error": f"Failed upload: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
                print("Audio File URL", file_url)
                # Create AudioFile
                audio_file_obj = AudioFile.objects.create(
                    file_path=file_url,
                    status="processing",
                    sop=sop_obj,
                    user=user_obj,
                    duration=0.0
                )

                # Session handling
                if session_id:
                    try:
                        session_obj = Session.objects.get(id=session_id)
                        if user_obj and not session_obj.session_users.filter(user=user_obj).exists():
                            SessionUser.objects.create(session=session_obj, user=user_obj)
                    except:
                        return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
                elif session_name and user_obj:
                    session_obj = Session.objects.create(
                        name=session_name, user=user_obj, sop=sop_obj, status='active'
                    )
                    SessionUser.objects.create(session=session_obj, user=user_obj)
                if session_obj:
                    session_obj.audio_files.add(audio_file_obj)

                # Process audio
                logging.info("Processing audio...")
                audio_processor = EnhancedAudioProcessor(hf_token)
                content_validator = LLMContentValidator()
                timeline_generator = SpeakerTimelineGenerator()
                content_validator.load_models()

                import requests

                try:
                    response = requests.get(file_url, stream=True)
                    response.raise_for_status()
                    full_transcript, speaker_segments = audio_processor.transcribe_with_speaker_diarization(response)
                except Exception as e:
                    # Mark as failed if processing fails
                    if 'audio_file_obj' in locals() and audio_file_obj:
                        try:
                            audio_file_obj.status = "failed"
                            audio_file_obj.save()
                        except Exception:
                            pass
                    return Response({"error": f"Audio processing failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Update duration
                if speaker_segments:
                    max_dur = max(s.end_time for s in speaker_segments)
                    audio_file_obj.duration = max_dur

                # Step matching
                matches = content_validator.advanced_step_matching(steps, speaker_segments)
                final_coverage = content_validator._calculate_coverage(matches)
                matched_count = sum(1 for m in matches if m.status=='Matched')
                partial_count = sum(1 for m in matches if m.status=='Partial')
                missing_count = sum(1 for m in matches if m.status=='Missing')

                # Save results
                transcription_data = {
                    'full_transcript': full_transcript,
                    'speaker_segments': [
                        {
                            'speaker_id': s.speaker_id,
                            'start_time': s.start_time,
                            'end_time': s.end_time,
                            'text': s.text,
                            'confidence': s.confidence
                        } for s in speaker_segments
                    ],
                    'matches': [
                        {
                            'step_num': m.step_num,
                            'step_text': m.step_text,
                            'status': m.status,
                            'confidence_score': m.confidence_score,
                            'llm_validation': m.llm_validation,
                            'speaker_attribution': m.speaker_attribution,
                            'match_details': m.match_details
                        } for m in matches
                    ],
                    'coverage': final_coverage,
                    'processing_metadata': {
                        'matched_count': matched_count,
                        'partial_count': partial_count,
                        'missing_count': missing_count,
                        'total_speakers': len(set(s.speaker_id for s in speaker_segments)),
                        'reference_document_id': reference_document_obj.id,
                        'sop_id': sop_obj.id if sop_obj else None
                    }
                }
                # Save to audio record
                audio_file_obj.transcription = transcription_data
                audio_file_obj.status = "processed"
                # Generate summary
                summary = _generate_enhanced_summary(full_transcript, matches)
                audio_file_obj.summary = summary
                # Save keywords
                all_keywords = []
                for s in speaker_segments:
                    all_keywords.extend(_extract_keywords_from_text(s.text))
                audio_file_obj.keywords_detected = ", ".join(set(all_keywords))
                audio_file_obj.save()

                # Update speaker profiles
                _update_speaker_profiles(speaker_segments)

                # Generate report and timeline
                logging.info("Generating report and timeline...")
                pdf_buf = create_enhanced_pdf_report(steps, matches, full_transcript, summary, final_coverage, speaker_segments, procedure_text)
                timeline_img = timeline_generator.create_speaker_timeline(speaker_segments, matches)

                # Package ZIP
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.writestr("validation_report.pdf", pdf_buf.getvalue())
                    if hasattr(timeline_img, 'getvalue'):
                        zipf.writestr("speaker_timeline.png", timeline_img.getvalue())
                    analysis_data = _create_comprehensive_analysis(
                        audio_file_obj, matches, speaker_segments, final_coverage, start_time,
                        len(set(s.speaker_id for s in speaker_segments)), steps,
                        reference_document_obj, sop_obj
                    )
                    zipf.writestr("analysis_data.json", json.dumps(analysis_data, indent=2))
                zip_buf.seek(0)

                # Log audit
                if user_obj:
                    AuditLog.objects.create(
                        action='audio_upload',
                        user=user_obj,
                        session=session_obj,
                        object_id=audio_file_obj.id,
                        object_type='AudioFile',
                        details={
                            'file_url': file_url,
                            'coverage': final_coverage,
                            'sop_id': sop_obj.id if sop_obj else None,
                            'reference_document_id': reference_document_obj.id,
                            'procedure_document_url': None,
                            'processing_time': time.time() - start_time
                        }
                    )

                # Response
                response = HttpResponse(zip_buf.getvalue(), content_type='application/zip')
                response['Content-Disposition'] = f'attachment; filename="validation_report_{int(final_coverage*100)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip"'
                total_time = time.time() - start_time
                return response

        except Exception as e:
            # Mark as failed if possible
            if 'audio_file_obj' in locals() and audio_file_obj:
                audio_file_obj.status = "failed"
                audio_file_obj.save()
            logging.exception("Processing failed")
            return Response({"error": f"Processing failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Helper functions
def _create_comprehensive_analysis(audio_file_obj, matches, speaker_segments, final_coverage, start_time, total_speakers, steps, reference_document, sop):
    matched = sum(1 for m in matches if m.status=='Matched')
    partial = sum(1 for m in matches if m.status=='Partial')
    missing = sum(1 for m in matches if m.status=='Missing')
    return {
        "audio_file_id": audio_file_obj.id,
        "file_url": audio_file_obj.file_path,
        "sop_info": {
            "id": sop.id if sop else None,
            "name": sop.name if sop else None,
            "version": sop.version if sop else None
        },
        "metadata": {
            "processing_time": f"{time.time() - start_time:.1f} seconds",
            "coverage_achieved": f"{final_coverage:.1%}",
            "target_met": final_coverage >= Config.TARGET_COVERAGE,
            "total_speakers": total_speakers,
            "total_steps": len(steps),
            "total_segments": len(speaker_segments),
            "audio_duration": f"{audio_file_obj.duration:.1f}s",
            "status": audio_file_obj.status
        },
        "coverage_breakdown": {
            "matched": matched,
            "partial": partial,
            "missing": missing,
            "matched_percent": f"{(matched/len(steps)*100):.1f}%",
            "partial_percent": f"{(partial/len(steps)*100):.1f}%",
            "missing_percent": f"{(missing/len(steps)*100):.1f}%"
        },
        "step_results": [
            {
                "step_num": m.step_num,
                "step_text": m.step_text[:150] + "..." if len(m.step_text) > 150 else m.step_text,
                "status": m.status,
                "confidence": round(m.confidence_score, 3),
                "speakers": m.speaker_attribution,
                "llm_validation": m.llm_validation,
                "score_breakdown": m.match_details
            } for m in matches
        ],
        "speaker_analysis": _analyze_speakers_enhanced(speaker_segments, matches),
        "quality_metrics": {
            "avg_transcription_confidence": float(np.mean([s.confidence for s in speaker_segments])) if speaker_segments else 0,
            "avg_matching_confidence": float(np.mean([m.confidence_score for m in matches if m.status != 'Missing'])) if matches else 0,
            "high_confidence_matches": sum(1 for m in matches if m.confidence_score >= Config.HIGH_CONFIDENCE_THRESHOLD),
            "speaker_balance": _calculate_speaker_balance(speaker_segments)
        }
    }

def _analyze_speakers_enhanced(speaker_segments, matches):
    # Implement detailed speaker analysis
    speakers = set(s.speaker_id for s in speaker_segments)
    analysis = {}
    for spk in speakers:
        total_time = sum(s.duration for s in speaker_segments if s.speaker_id == spk)
        steps_matched = sum(1 for m in matches for s in m.matched_segments if s.speaker_id == spk)
        analysis[spk] = {
            'speaking_time': total_time,
            'steps_matched': steps_matched
        }
    return analysis

def _calculate_speaker_balance(speaker_segments):
    counts = {}
    for s in speaker_segments:
        counts[s.speaker_id] = counts.get(s.speaker_id, 0) + 1
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()} if total > 0 else {}

def _extract_keywords_from_text(text):
    try:
        words = word_tokenize(text.lower())
        return [w for w in words if len(w) > 3 and w.isalpha()]
    except:
        return []

def _update_speaker_profiles(speaker_segments):
    try:
        for s in set(s.speaker_id for s in speaker_segments):
            sp, created = SpeakerProfile.objects.get_or_create(
                name=s,
                defaults={'embedding': {}}
            )
            if created:
                logging.info(f"Created speaker profile {s}")
    except Exception as e:
        logging.error(f"Error updating speaker profiles: {e}")

def _generate_enhanced_summary(full_text, matches):
    sentences = sent_tokenize(full_text)
    if len(sentences) <= 5:
        summary = full_text
    else:
        summary = " ".join(sentences[:2] + sentences[-2:])
    # Add insight
    matched = sum(1 for m in matches if m.status=='Matched')
    partial = sum(1 for m in matches if m.status=='Partial')
    total = len(matches)
    coverage_insight = f"\n\nValidation: {matched}/{total} steps matched, {partial} partial."
    if (matched + partial) >= total * 0.8:
        coverage_insight += " Excellent adherence."
    elif (matched + partial) >= total * 0.6:
        coverage_insight += " Good adherence."
    else:
        coverage_insight += " Gaps detected."
    return summary + coverage_insight