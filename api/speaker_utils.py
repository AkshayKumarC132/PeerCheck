from typing import List, Dict, Optional

import numpy as np
from numpy.linalg import norm

from .models import SpeakerProfile


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.isnan(a).any() or np.isnan(b).any() or np.isinf(a).any() or np.isinf(b).any():
        return 0.0
    denom = norm(a) * norm(b)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(np.dot(a, b) / denom)


def match_speaker_embedding(embedding: List[float], threshold: float = 0.8) -> Optional[SpeakerProfile]:
    """Return the best matching speaker profile or None."""
    best_profile = None
    best_score = 0.0
    for profile in SpeakerProfile.objects.all():
        profile_embedding = profile.embedding or []
        score = _cosine_similarity(embedding, profile_embedding)
        if score > best_score:
            best_score = score
            best_profile = profile
    if best_profile and best_score >= threshold:
        return best_profile
    return None


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
        arr = np.array(vecs, dtype=float)
        if arr.size == 0 or np.isnan(arr).any() or np.isinf(arr).any():
            continue
        mean_vec = np.mean(arr, axis=0)
        if mean_vec is None or np.isnan(mean_vec).any() or np.isinf(mean_vec).any():
            continue
        mean_vec_list = mean_vec.tolist()
        profile = match_speaker_embedding(mean_vec_list, threshold)
        if profile is None:
            profile = SpeakerProfile.objects.create(embedding=mean_vec_list, name=name)
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

