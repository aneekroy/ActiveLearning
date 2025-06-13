"""Wrapper utilities around HuggingFace causal language models."""

from typing import List
from pathlib import Path

import os


# Prevent transformers from importing TensorFlow or Flax which slows down
# startup and emits unwanted log messages.
os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORT", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


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
                
            torch.backends.cudnn.benchmark = True

        self.model = self.model.to(device).eval()




    def compute_perplexity(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return math.exp(loss.item())

    def compute_batch_perplexity(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """Compute perplexities for a list of texts in batches."""
        perplexities = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True)
            labels = enc["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc, labels=labels.to(self.device)).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].to(self.device)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            losses = losses.view(shift_labels.size())
            mask = shift_labels != -100
            seq_loss = (losses * mask).sum(1) / mask.sum(1)
            perplexities.extend(torch.exp(seq_loss).tolist())
        return perplexities

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
