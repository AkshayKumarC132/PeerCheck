import re
from typing import List, Dict

CONFIRMATION_PHRASES = [
    "that's right", "that's correct", "exactly", "correct me if i'm wrong"
]

class TranscriptValidator:
    def confirm_phrases(self, text: str) -> List[str]:
        found = []
        lower = text.lower()
        for phrase in CONFIRMATION_PHRASES:
            if phrase in lower:
                found.append(phrase)
        return found

    def detect_repetition(self, text: str) -> List[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        repeats = []
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                repeats.append(tokens[i])

        phrase_pattern = re.compile(r"(\b\w+\b(?:\s+\b\w+\b)+)")
        for match in phrase_pattern.finditer(text.lower()):
            phrase = match.group(1)
            words = phrase.split()
            mid = len(words) // 2
            if words[:mid] == words[mid:]:
                repeats.append(" ".join(words[:mid]))

        return list(dict.fromkeys(repeats))

    def structural_coverage(self, matches: List[Dict], document_sections: List[str]) -> Dict[str, List[str]]:
        matched_sections = {m["matched_section"] for m in matches if m.get("status") != "missing"}
        missing = [s for s in document_sections if s not in matched_sections]
        return {"missing_sections": missing}
