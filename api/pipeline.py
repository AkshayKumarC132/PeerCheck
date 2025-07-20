"""Async processing pipeline for PeerCheck."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional

import logging
import requests

from .new_enhanced import (
    EnhancedAudioProcessor,
    LLMContentValidator,
    SpeakerTimelineGenerator,
    SpeakerSegment,
    StepMatch,
    _generate_enhanced_summary,
)

@dataclass
class TranscriptMetadata:
    """Container for enriched transcript data."""

    transcript: str
    segments: List[SpeakerSegment]
    matches: List[StepMatch]
    coverage: float
    summary: str
    timeline: bytes


class PeerCheckPipeline:
    """High level pipeline orchestrating the PeerCheck workflow."""

    def __init__(self, hf_token: Optional[str] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.audio_processor = EnhancedAudioProcessor(hf_token)
        self.validator = LLMContentValidator()
        self.validator.load_models()
        self.timeline_generator = SpeakerTimelineGenerator()

    async def process(
        self,
        audio_url: str,
        steps: List[Tuple[str, str]],
        procedure_text: str,
    ) -> TranscriptMetadata:
        loop = asyncio.get_running_loop()
        self.logger.info("Starting pipeline for %s", audio_url)
        try:
            self.logger.info("Fetching audio file")
            response = await asyncio.to_thread(
                requests.get, audio_url, stream=True, timeout=30
            )
            response.raise_for_status()
            self.logger.info("Audio fetched successfully")
        except requests.RequestException as exc:
            self.logger.error("Failed to download audio: %s", exc)
            raise RuntimeError(f"Failed to fetch audio: {exc}") from exc

        from contextlib import closing

        with closing(response):
            self.logger.info("Transcribing and diarizing audio")
            transcript, segments = await asyncio.to_thread(
                self.audio_processor.transcribe_with_speaker_diarization, response
            )
        self.logger.info("Transcription produced %d segments", len(segments))

        self.logger.info("Running step matching")
        matches = await asyncio.to_thread(
            self.validator.advanced_step_matching, steps, segments
        )
        coverage = self.validator._calculate_coverage(matches)
        summary = _generate_enhanced_summary(transcript, matches)
        self.logger.info("Generating timeline and summary")
        timeline_buf = self.timeline_generator.create_speaker_timeline(segments, matches)
        timeline_bytes = timeline_buf.getvalue() if hasattr(timeline_buf, "getvalue") else b""

        result = TranscriptMetadata(
            transcript=transcript,
            segments=segments,
            matches=matches,
            coverage=coverage,
            summary=summary,
            timeline=timeline_bytes,
        )
        self.logger.info("Pipeline completed with coverage %.2f", coverage)
        return result

