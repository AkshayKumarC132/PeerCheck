from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

class EmbeddingMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def _split_document(self, text: str) -> List[str]:
        sections = re.split(r"\n\s*\n", text)
        return [s.strip() for s in sections if s.strip()]

    def embed(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def match(self, transcript: str, document: str, threshold: float = 0.7) -> List[Dict]:
        doc_sections = self._split_document(document)
        doc_embeds = self.embed(doc_sections)
        index = faiss.IndexFlatIP(doc_embeds.shape[1])
        faiss.normalize_L2(doc_embeds)
        index.add(doc_embeds)

        trans_segments = self._split_document(transcript)
        trans_embeds = self.embed(trans_segments)
        faiss.normalize_L2(trans_embeds)
        D, I = index.search(trans_embeds, 1)

        matches = []
        for i, seg in enumerate(trans_segments):
            score = float(D[i][0])
            idx = int(I[i][0])
            matched = doc_sections[idx] if idx < len(doc_sections) else ""
            if score >= threshold:
                status = "matched"
            elif score >= threshold * 0.5:
                status = "partial"
            else:
                status = "missing"
                matched = ""
            matches.append({
                "segment": seg,
                "matched_section": matched,
                "score": score,
                "status": status,
            })
        return matches
