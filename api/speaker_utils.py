import numpy as np
from numpy.linalg import norm
from typing import List, Dict, Optional, Tuple

from .models import SpeakerProfile


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def find_best_speaker_profile(
    embedding: List[float], threshold: float = 0.8
) -> Tuple[Optional[SpeakerProfile], float]:
    """Return the best matching speaker profile and cosine similarity score."""
    best_profile = None
    best_score = 0.0
    for profile in SpeakerProfile.objects.all():
        score = _cosine_similarity(embedding, profile.embedding)
        if score > best_score:
            best_score = score
            best_profile = profile
    if best_profile and best_score >= threshold:
        return best_profile, best_score
    return None, best_score


def match_speaker_embedding(
    embedding: List[float], threshold: float = 0.8
) -> Optional[SpeakerProfile]:
    """Return the best matching speaker profile or ``None``."""
    profile, _ = find_best_speaker_profile(embedding, threshold)
    return profile


def assign_speaker_profiles(
    segments: List[Dict],
    threshold: float = 0.8,
    allowed_speakers: Optional[List[str]] = None,
) -> List[Dict]:
    """Attach speaker profiles to segments.

    When ``allowed_speakers`` is provided, new profiles are only created for
    those speaker names. This prevents saving embeddings for spurious speakers
    that the diarization step may have introduced.
    """

    # Aggregate vectors by speaker label so each speaker generates at most one
    # profile. This also allows averaging vectors from multiple segments.
    speaker_vectors: Dict[str, List[List[float]]] = {}
    for seg in segments:
        vec = seg.get("speaker_vector")
        name = seg.get("speaker")
        if vec is not None and name:
            speaker_vectors.setdefault(name, []).append(vec)

    profiles: Dict[str, SpeakerProfile] = {}
    for name, vecs in speaker_vectors.items():
        if allowed_speakers is not None and name not in allowed_speakers:
            continue
        mean_vec = np.mean(np.array(vecs, dtype=float), axis=0).tolist()
        profile = match_speaker_embedding(mean_vec, threshold)
        if profile is None:
            profile = SpeakerProfile.objects.create(embedding=mean_vec, name=name)
        profiles[name] = profile

    updated_segments: List[Dict] = []
    for seg in segments:
        name = seg.get("speaker")
        profile = profiles.get(name)
        if profile:
            seg["speaker_profile_id"] = profile.id
            seg["speaker"] = profile.name or name
        updated_segments.append(seg)
    return updated_segments

