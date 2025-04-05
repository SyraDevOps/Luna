from typing import List, Dict
import torch
import numpy as np

class Evaluator:
    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizerFast, config: Config) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def evaluate(self, texts: List[str], references: List[str]) -> Dict[str, float]:
        results = {}
        if self.config.eval_metrics.compute_bleu:
            preds = [self.tokenizer.decode(self.model.generate(
                self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.data.max_seq_length).to(DEVICE),
                max_new_tokens=100)[0], skip_special_tokens=True) for text in texts]
            results["bleu"] = compute_bleu(preds, references)
        if self.config.eval_metrics.compute_perplexity:
            results["perplexity"] = compute_perplexity(self.model, self.tokenizer, texts)
        results["distinct-2"] = compute_distinct_n(preds, n=2)
        return results