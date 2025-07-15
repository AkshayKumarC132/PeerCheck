from typing import List, Dict
from .validators import TranscriptValidator

class ReportGenerator:
    def __init__(self):
        self.validator = TranscriptValidator()

    def create_report(
        self,
        transcript: str,
        matches: List[Dict],
        document_sections: List[str],
        llm_summary: str,
        llm_sections: List[Dict],
    ) -> Dict:
        validation = {
            "confirm_phrases": self.validator.confirm_phrases(transcript),
            "repetition": self.validator.detect_repetition(transcript),
            "structural": self.validator.structural_coverage(matches, document_sections),
        }
        return {
            "transcript": transcript,
            "matches": matches,
            "validation": validation,
            "summary": llm_summary,
            "section_scores": llm_sections,
        }
