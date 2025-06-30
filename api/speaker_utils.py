import numpy as np
from numpy.linalg import norm
from typing import List, Dict

from .models import SpeakerProfile


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def match_speaker_embedding(embedding: List[float], threshold: float = 0.8) -> SpeakerProfile | None:
    """Return the best matching speaker profile or None."""
    best_profile = None
    best_score = 0.0
    for profile in SpeakerProfile.objects.all():
        score = _cosine_similarity(embedding, profile.embedding)
        if score > best_score:
            best_score = score
            best_profile = profile
    if best_profile and best_score >= threshold:
        return best_profile
    return None


def assign_speaker_profiles(segments: List[Dict], threshold: float = 0.8) -> List[Dict]:
    """Attach speaker profiles to transcription segments and create new profiles if needed."""
    updated_segments = []
    for seg in segments:
        vector = seg.get("speaker_vector")
        if vector is None:
            updated_segments.append(seg)
            continue
        profile = match_speaker_embedding(vector, threshold)
        if profile is None:
            profile = SpeakerProfile.objects.create(embedding=vector)
        seg["speaker_profile_id"] = profile.id
        seg["speaker"] = profile.name or seg.get("speaker")
        updated_segments.append(seg)
    return updated_segments

