import torch
import logging
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizerFast
import random
import json

logger = logging.getLogger(__name__)

class SupervisedDataset(Dataset):
    """Dataset para treinamento supervisionado do LunaGPT"""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: PreTrainedTokenizerFast, 
        max_length: int = 512,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True
    ):
        """
        Inicializa o dataset supervisionado
        
        Args:
            texts: Lista de textos para treinamento
            tokenizer: Tokenizer para processar os textos
            max_length: Comprimento máximo das sequências
            add_special_tokens: Se deve adicionar tokens especiais
            return_attention_mask: Se deve retornar máscara de atenção
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.return_attention_mask = return_attention_mask
        
        # Pré-processar textos
        self.processed_data = self._preprocess_texts()
        
        logger.info(f"Dataset supervisionado criado com {len(self.texts)} amostras")
    
    def _preprocess_texts(self) -> List[Dict[str, torch.Tensor]]:
        """Pré-processa todos os textos"""
        processed_data = []
        
        for i, text in enumerate(self.texts):
            try:
                # Tokenizar texto
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                    add_special_tokens=self.add_special_tokens,
                    return_attention_mask=self.return_attention_mask
                )
                
                # Preparar dados para o formato esperado
                item = {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "labels": encoded["input_ids"].squeeze(0).clone()
                }
                
                if self.return_attention_mask:
                    item["attention_mask"] = encoded["attention_mask"].squeeze(0)
                
                processed_data.append(item)
                
            except Exception as e:
                logger.error(f"Erro ao processar texto {i}: {e}")
        
        if not processed_data:
            logger.warning("Nenhum texto foi processado com sucesso")
        
        return processed_data
    
    def __len__(self) -> int:
        """Retorna o tamanho do dataset"""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retorna um item do dataset
        
        Args:
            idx: Índice do item
            
        Returns:
            Dicionário com input_ids, attention_mask e labels
        """
        if idx >= len(self.processed_data):
            raise IndexError(f"Índice {idx} fora do alcance. Dataset tem {len(self.processed_data)} itens.")
        
        return self.processed_data[idx]
    
    def get_text(self, idx: int) -> str:
        """
        Retorna o texto original de um índice
        
        Args:
            idx: Índice do texto
            
        Returns:
            Texto original
        """
        if idx >= len(self.texts):
            raise IndexError(f"Índice {idx} fora do alcance. Dataset tem {len(self.texts)} textos.")
        
        return self.texts[idx]
    
    def add_texts(self, new_texts: List[str]):
        """
        Adiciona novos textos ao dataset
        
        Args:
            new_texts: Lista de novos textos
        """
        self.texts.extend(new_texts)
        
        # Processar novos textos
        new_processed = []
        for text in new_texts:
            try:
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                    add_special_tokens=self.add_special_tokens,
                    return_attention_mask=self.return_attention_mask
                )
                
                item = {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "labels": encoded["input_ids"].squeeze(0).clone()
                }
                
                if self.return_attention_mask:
                    item["attention_mask"] = encoded["attention_mask"].squeeze(0)
                
                new_processed.append(item)
                
            except Exception as e:
                logger.error(f"Erro ao processar novo texto: {e}")
        
        self.processed_data.extend(new_processed)
        logger.info(f"Adicionados {len(new_texts)} novos textos ao dataset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do dataset
        
        Returns:
            Dicionário com estatísticas
        """
        if not self.processed_data:
            return {"total_samples": 0, "avg_length": 0, "max_length": 0, "min_length": 0}
        
        lengths = []
        for item in self.processed_data:
            # Calcular comprimento real (sem padding)
            input_ids = item["input_ids"]
            if self.return_attention_mask:
                length = item["attention_mask"].sum().item()
            else:
                # Contar tokens não-pad
                length = (input_ids != self.tokenizer.pad_token_id).sum().item()
            lengths.append(length)
        
        return {
            "total_samples": len(self.processed_data),
            "avg_length": sum(lengths) / len(lengths),
            "max_length": max(lengths),
            "min_length": min(lengths),
            "vocab_size": self.tokenizer.vocab_size
        }
    
    def save_to_file(self, filepath: str):
        """
        Salva o dataset em um arquivo
        
        Args:
            filepath: Caminho do arquivo para salvar
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.texts, f, ensure_ascii=False, indent=2)
            logger.info(f"Dataset salvo em {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar dataset: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str, tokenizer: PreTrainedTokenizerFast, **kwargs):
        """
        Carrega dataset de um arquivo
        
        Args:
            filepath: Caminho do arquivo
            tokenizer: Tokenizer para processar os textos
            **kwargs: Argumentos adicionais para o construtor
            
        Returns:
            Instância do SupervisedDataset
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            logger.info(f"Dataset carregado de {filepath}")
            return cls(texts, tokenizer, **kwargs)
        except Exception as e:
            logger.error(f"Erro ao carregar dataset: {e}")
            return cls([], tokenizer, **kwargs)
    
    def filter_by_length(self, min_length: int = 0, max_length: Optional[int] = None):
        """
        Filtra o dataset por comprimento
        
        Args:
            min_length: Comprimento mínimo
            max_length: Comprimento máximo (opcional)
        """
        filtered_texts = []
        for text in self.texts:
            length = len(self.tokenizer.encode(text))
            if length >= min_length:
                if max_length is None or length <= max_length:
                    filtered_texts.append(text)
        
        self.texts = filtered_texts
        self.processed_data = self._preprocess_texts()
        logger.info(f"Dataset filtrado: {len(self.texts)} textos restantes")
    
    def shuffle(self):
        """Embaralha o dataset"""
        combined = list(zip(self.texts, self.processed_data))
        random.shuffle(combined)
        self.texts, self.processed_data = zip(*combined)
        self.texts = list(self.texts)
        self.processed_data = list(self.processed_data)
        logger.info("Dataset embaralhado")
    
    def split(self, train_ratio: float = 0.8) -> tuple['SupervisedDataset', 'SupervisedDataset']:
        """
        Divide o dataset em treino e validação
        
        Args:
            train_ratio: Proporção para treino
            
        Returns:
            Tupla com datasets de treino e validação
        """
        split_idx = int(len(self.texts) * train_ratio)
        
        train_texts = self.texts[:split_idx]
        val_texts = self.texts[split_idx:]
        
        train_dataset = SupervisedDataset(
            train_texts, 
            self.tokenizer, 
            self.max_length, 
            self.add_special_tokens, 
            self.return_attention_mask
        )
        
        val_dataset = SupervisedDataset(
            val_texts, 
            self.tokenizer, 
            self.max_length, 
            self.add_special_tokens, 
            self.return_attention_mask
        )
        
        logger.info(f"Dataset dividido: {len(train_texts)} treino, {len(val_texts)} validação")
        return train_dataset, val_dataset