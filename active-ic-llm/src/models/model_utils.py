"""Wrapper utilities around HuggingFace causal language models."""

from typing import List
from pathlib import Path

import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math


class ModelUtils:
    def __init__(self, model_name: str, device: str = "cpu", num_gpus: int = 1):
        self.device = device
        self.num_gpus = num_gpus

        # If the user provides a local path, avoid any network downloads by
        # forcing HuggingFace to load files only from that directory.
        local = Path(model_name).exists()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local
        )

        if device.startswith("cuda") and torch.cuda.is_available():
            available = torch.cuda.device_count()
            if available > 1:
                gpus = list(range(min(self.num_gpus, available)))
                self.model = torch.nn.DataParallel(self.model, device_ids=gpus)

        self.model = self.model.to(device)


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
