"""Wrapper utilities around HuggingFace causal language models."""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_HAS_UNSLOTH = False
FastLanguageModel = None
import math


class ModelUtils:
    def __init__(self, model_name: str, device: str = "cpu", use_unsloth: bool = False):
        self.device = device
        if use_unsloth:
            global _HAS_UNSLOTH, FastLanguageModel
            if not _HAS_UNSLOTH:
                try:
                    from unsloth.models import FastLanguageModel as _FLM
                    FastLanguageModel = _FLM
                    _HAS_UNSLOTH = True
                except Exception:
                    _HAS_UNSLOTH = False
            if _HAS_UNSLOTH:
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name,
                    device_map="auto",
                    use_gradient_checkpointing=False,
                )
                self.model = self.model.to(device)
                return
        # fall back to standard transformers loader
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def compute_perplexity(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return math.exp(loss.item())

    def _score_completion(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        ids = inputs["input_ids"][:, 1:]
        token_log_prob = log_probs.gather(2, ids.unsqueeze(-1)).squeeze(-1).sum().item()
        return token_log_prob

    def predict_classification(self, prompt: str, candidate_labels: List[str]) -> str:
        scores = {label: self._score_completion(prompt + " " + label) for label in candidate_labels}
        return max(scores, key=scores.get)

    def predict_multichoice(self, prompt: str, choices: List[str]) -> str:
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[: len(choices)]
        scores = {}
        for letter, choice in zip(letters, choices):
            scores[letter] = self._score_completion(prompt + f" {letter}) {choice}")
        return max(scores, key=scores.get)
