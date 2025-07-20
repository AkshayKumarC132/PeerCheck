"""Async processing pipeline for PeerCheck."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
        try:
            response = await asyncio.to_thread(
                requests.get, audio_url, stream=True, timeout=30
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to fetch audio: {exc}") from exc

        from contextlib import closing

        with closing(response):
            transcript, segments = await asyncio.to_thread(
                self.audio_processor.transcribe_with_speaker_diarization, response
            )

        matches = await asyncio.to_thread(
            self.validator.advanced_step_matching, steps, segments
        )
        coverage = self.validator._calculate_coverage(matches)
        summary = _generate_enhanced_summary(transcript, matches)
        timeline_buf = self.timeline_generator.create_speaker_timeline(segments, matches)
        timeline_bytes = timeline_buf.getvalue() if hasattr(timeline_buf, "getvalue") else b""

        return TranscriptMetadata(
            transcript=transcript,
            segments=segments,
            matches=matches,
            coverage=coverage,
            summary=summary,
            timeline=timeline_bytes,
        )

