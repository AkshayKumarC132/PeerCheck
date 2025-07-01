import numpy as np
from numpy.linalg import norm
from typing import List, Dict, Optional

from .models import SpeakerProfile


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def match_speaker_embedding(embedding: List[float], threshold: float = 0.8) -> Optional[SpeakerProfile]:
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


def apply_custom_speaker_names(
    audio_instance,
    name_mapping: Dict[str, str],
    threshold: float = 0.8,
) -> List[Dict]:
    """Update transcription speaker labels with custom names.

    For each ``old_label`` -> ``new_name`` pair in ``name_mapping`` this will:
    - Aggregate the speaker vectors for ``old_label`` across the transcription.
    - Find or create a :class:`SpeakerProfile` representing that speaker.
    - Update the profile name to ``new_name``.
    - Replace the ``speaker`` field and ``speaker_profile_id`` in the
      transcription segments.

    Parameters
    ----------
    audio_instance: AudioFile
        The audio object whose transcription should be updated.
    name_mapping: Dict[str, str]
        Mapping of existing speaker labels (e.g. ``"Speaker_1"``) to the desired
        custom names.
    threshold: float
        Similarity threshold when searching for an existing profile.

    Returns
    -------
    List[Dict]
        The updated transcription segments.
    """

    transcription = audio_instance.transcription or []
    if not isinstance(transcription, list):
        return transcription

    # Collect vectors for each speaker label we are renaming
    label_vectors: Dict[str, List[List[float]]] = {k: [] for k in name_mapping}
    for seg in transcription:
        label = seg.get("speaker")
        if label in name_mapping and isinstance(seg.get("speaker_vector"), list):
            label_vectors[label].append(seg["speaker_vector"])

    profiles: Dict[str, SpeakerProfile] = {}
    for old_label, new_name in name_mapping.items():
        vectors = label_vectors.get(old_label, [])
        profile = None

        if vectors:
            mean_vec = np.mean(np.array(vectors, dtype=float), axis=0).tolist()
            profile = match_speaker_embedding(mean_vec, threshold)
            if profile is None:
                profile = SpeakerProfile.objects.create(embedding=mean_vec, name=new_name)
            else:
                if profile.name != new_name:
                    profile.name = new_name
                    profile.save()
        else:
            # Fallback to existing profile id if present
            for seg in transcription:
                if seg.get("speaker") == old_label and seg.get("speaker_profile_id"):
                    try:
                        profile = SpeakerProfile.objects.get(id=seg["speaker_profile_id"])
                        if profile.name != new_name:
                            profile.name = new_name
                            profile.save()
                    except SpeakerProfile.DoesNotExist:
                        profile = None
                    break
            if profile is None:
                profile = SpeakerProfile.objects.create(name=new_name, embedding=[])

        profiles[old_label] = profile

    # Apply updates to the transcription
    for seg in transcription:
        label = seg.get("speaker")
        if label in name_mapping:
            profile = profiles[label]
            seg["speaker"] = name_mapping[label]
            seg["speaker_profile_id"] = profile.id

    audio_instance.transcription = transcription
    audio_instance.save()
    return transcription

