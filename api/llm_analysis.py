from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LlamaAnalyzer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _generate(self, prompt: str, max_tokens: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def summarize(self, transcript: str) -> str:
        prompt = "Summarize the following transcript:\n" + transcript
        return self._generate(prompt)

    def score_sections(self, transcript: str, sections: List[str]) -> List[Dict]:
        results = []
        for sec in sections:
            prompt = (
                "Rate the compliance of the following transcript to the reference section on a scale of 1 to 5 and provide short feedback.\n"
                f"Section: {sec}\nTranscript: {transcript}\nReturn as JSON with keys score and feedback."
            )
            resp = self._generate(prompt)
            results.append({"section": sec, "analysis": resp})
        return results
