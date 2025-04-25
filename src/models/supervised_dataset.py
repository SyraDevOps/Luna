import torch
from torch.utils.data import Dataset

class SupervisedDataset(Dataset):
    """Dataset para treinamento supervisionado"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = []
        
        for text in texts:
            # Tokenizar o texto
            tokenized = tokenizer(
                text, 
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Armazenar entrada tokenizada
            self.inputs.append({
                "input_ids": tokenized["input_ids"].squeeze(),
                "attention_mask": tokenized["attention_mask"].squeeze(),
                "labels": tokenized["input_ids"].squeeze()  # Para modelagem causal, os labels são iguais aos input_ids
            })
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]