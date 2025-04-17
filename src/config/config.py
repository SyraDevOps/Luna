import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass

class ModelConfig:
    def __init__(self):
        self.vocab_size = 50257
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.n_positions = 1024  # Adicionado n_positions
        self.use_lightweight_mode = True  # Controla otimização para hardware leve

class TrainingConfig:
    def __init__(self):
        self.num_train_epochs = 5
        self.per_device_train_batch_size = 2
        self.per_device_eval_batch_size = 2
        self.logging_steps = 10
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.gradient_checkpointing = False
        self.use_wandb = True  # Alterado para True para habilitar por padrão

@dataclass
class PersonaConfig:
    """Configuração de personas do chat"""
    default: str = "casual"
    options: List[str] = field(default_factory=lambda: ["tecnico", "casual", "formal"])

class FeedbackConfig:
    def __init__(self):
        # Existing attributes
        self.feedback_file = "feedback.jsonl"
        self.min_samples_for_update = 10
        self.quality_threshold = 4  # Add this default threshold for high-quality feedback

class Config:
    def __init__(self, config_path=None):
        # Initialize configs with default values
        self.model = ModelConfig()
        self.training = TrainingConfig() 
        self.persona = PersonaConfig()
        self.feedback = FeedbackConfig()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
            
        # Garantir que a validação seja sempre chamada no final do construtor
        self.validate()
    
    def load_from_file(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Update configs from loaded data
            if "model" in data:
                for k, v in data["model"].items():
                    setattr(self.model, k, v)
            
            if "training" in data:
                for k, v in data["training"].items():
                    setattr(self.training, k, v)
            
            if "persona" in data:
                for k, v in data["persona"].items():
                    setattr(self.persona, k, v)
            
            if "feedback" in data:
                for k, v in data["feedback"].items():
                    setattr(self.feedback, k, v)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
    
    def validate(self):
        """Valida a configuração e aplica valores padrão quando necessário."""
        # Verificar e corrigir vocab_size
        if not hasattr(self.model, 'vocab_size'):
            self.model.vocab_size = 50000  # Valor padrão razoável
        elif hasattr(self.model, 'vocab_size') and (self.model.vocab_size <= 0 or self.model.vocab_size == -100):
            self.model.vocab_size = 50000  # Corrigir valor negativo
        
        # Outras validações...
        
        return self