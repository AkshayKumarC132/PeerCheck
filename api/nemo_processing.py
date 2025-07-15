from typing import List, Dict
import numpy as np
import nemo.collections.asr as nemo_asr
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering

class NemoASR:
    def __init__(self, asr_model: str = "stt_en_fastconformer", spk_model: str = "speakerverification_speakernet"):
        self.asr = nemo_asr.models.EncDecCTCModel.from_pretrained(asr_model)
        self.spk = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(spk_model)

    def transcribe_with_diarization(self, audio_path: str, chunk_len: float = 5.0) -> List[Dict]:
        """Transcribe audio with basic speaker diarization."""
        signal, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            import librosa
            signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        samples = int(chunk_len * sample_rate)
        chunks = [signal[i:i+samples] for i in range(0, len(signal), samples)]

        texts = [self.asr.transcribe([c], batch_size=1)[0] for c in chunks]
        embeddings = [self.spk.get_embedding([c])[0] for c in chunks]

        if len(embeddings) > 1:
            emb_np = np.stack(embeddings)
            clustering = AgglomerativeClustering(n_clusters=min(len(chunks), 2), metric="cosine", linkage="average")
            labels = clustering.fit_predict(emb_np)
        else:
            labels = [0] * len(embeddings)

        results = []
        ts = 0.0
        for text, label in zip(texts, labels):
            results.append({"speaker": f"Speaker_{label+1}", "text": text, "timestamp": ts})
            ts += chunk_len
        return results
