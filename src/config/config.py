import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuração do modelo Luna"""
    model_name: str = "luna_model"
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    
    # Configurações específicas do Luna
    use_moe: bool = True
    num_experts: int = 8
    top_k_experts: int = 2
    use_hypernet: bool = True
    use_growing_network: bool = True
    use_state_space: bool = True
    
    # Configurações de dropout
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    
    # Configurações de inicialização
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    
    # Configurações de memória
    gradient_checkpointing: bool = False
    use_cache: bool = True

@dataclass
class TrainingConfig:
    """Configuração de treinamento"""
    output_dir: str = "temp/training"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    
    # Configurações específicas
    use_wandb: bool = False
    gradient_checkpointing: bool = False
    fp16: bool = False
    dataloader_num_workers: int = 0
    
    # Configurações de curriculum learning
    use_curriculum: bool = False
    curriculum_stages: int = 3
    
    # Configurações de otimização
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

@dataclass
class FeedbackConfig:
    """Configuração do sistema de feedback"""
    feedback_file: str = "data/feedback.jsonl"
    memory_file: str = "data/memory.jsonl"
    quality_threshold: int = 4
    min_samples_for_update: int = 10
    collect_user_feedback: bool = True
    feedback_weight: float = 1.0

@dataclass
class RAGConfig:
    """Configuração do sistema RAG"""
    use_rag: bool = True
    index_path: str = "data/rag_index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 1024

@dataclass
class PersonaConfig:
    """Configuração das personas"""
    default_persona: str = "casual"
    available_personas: List[str] = field(default_factory=lambda: [
        "casual", "tecnico", "formal", "amigavel", "profissional"
    ])
    persona_config_file: str = "config/personas.json"

@dataclass
class MemoryConfig:
    """Configuração do sistema de memória"""
    memory_file: str = "data/memory.jsonl"
    max_memory_entries: int = 10000
    memory_retention_days: int = 30
    use_semantic_memory: bool = True
    memory_importance_threshold: float = 0.5

@dataclass
class OptimizationConfig:
    """Configuração de otimizações"""
    use_automl: bool = False
    automl_trials: int = 50
    use_dynamic_batching: bool = True
    use_mixed_precision: bool = False
    use_gradient_clipping: bool = True
    
    # Configurações de hardware
    auto_device_map: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False

class Config:
    """Classe principal de configuração"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa a configuração
        
        Args:
            config_path: Caminho para arquivo de configuração JSON (opcional)
        """
        # Configurações padrão
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.feedback = FeedbackConfig()
        self.rag = RAGConfig()
        self.persona = PersonaConfig()
        self.memory = MemoryConfig()
        self.optimization = OptimizationConfig()
        
        # Carregar configuração de arquivo se fornecido
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # Aplicar variáveis de ambiente
        self._apply_env_vars()
        
        # Validar configuração
        self._validate_config()
    
    def _load_from_file(self, config_path: str):
        """Carrega configuração de arquivo JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Atualizar configurações com dados do arquivo
            for section_name, section_data in config_data.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
                        else:
                            logger.warning(f"Configuração desconhecida: {section_name}.{key}")
                else:
                    logger.warning(f"Seção de configuração desconhecida: {section_name}")
            
            logger.info(f"Configuração carregada de {config_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar configuração de {config_path}: {e}")
    
    def _apply_env_vars(self):
        """Aplica variáveis de ambiente à configuração"""
        env_mappings = {
            "LUNA_LEARNING_RATE": ("training", "learning_rate", float),
            "LUNA_BATCH_SIZE": ("training", "per_device_train_batch_size", int),
            "LUNA_EPOCHS": ("training", "num_train_epochs", int),
            "LUNA_USE_WANDB": ("training", "use_wandb", lambda x: x.lower() == 'true'),
            "LUNA_MODEL_SIZE": ("model", "hidden_size", int),
            "LUNA_USE_RAG": ("rag", "use_rag", lambda x: x.lower() == 'true'),
            "LUNA_FEEDBACK_THRESHOLD": ("feedback", "quality_threshold", int),
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = converter(os.environ[env_var])
                    setattr(getattr(self, section), key, value)
                    logger.info(f"Configuração aplicada de variável de ambiente: {env_var}")
                except Exception as e:
                    logger.error(f"Erro ao aplicar variável de ambiente {env_var}: {e}")
    
    def _validate_config(self):
        """Valida a configuração e aplica valores padrão quando necessário."""
        # Se vocab_size for negativo ou zero, utilizar valor padrão
        if hasattr(self.model, 'vocab_size') and (not self.model.vocab_size or self.model.vocab_size <= 0):
            self.model.vocab_size = 5000  # Valor padrão seguro
        
        # Validar configurações do modelo
        if self.model.hidden_size % self.model.num_attention_heads != 0:
            logger.warning("hidden_size deve ser divisível por num_attention_heads")
        
        # Validar configurações de treinamento
        if self.training.learning_rate <= 0:
            logger.warning("learning_rate deve ser positivo")
        
        if self.training.per_device_train_batch_size <= 0:
            logger.warning("per_device_train_batch_size deve ser positivo")
        
        # Validar caminhos de arquivo
        paths_to_check = [
            self.feedback.feedback_file,
            self.feedback.memory_file,
            self.memory.memory_file
        ]
        
        for path in paths_to_check:
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Diretório criado: {dir_path}")
    
    def save_to_file(self, config_path: str):
        """Salva configuração atual em arquivo JSON"""
        try:
            config_dict = {}
            
            # Converter dataclasses para dicionários
            for attr_name in dir(self):
                if not attr_name.startswith('_'):
                    attr = getattr(self, attr_name)
                    if hasattr(attr, '__dict__'):
                        config_dict[attr_name] = attr.__dict__
            
            # Garantir que o diretório existe
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuração salva em {config_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar configuração em {config_path}: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna um resumo da configuração atual"""
        return {
            "model": {
                "name": self.model.model_name,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_hidden_layers,
                "use_moe": self.model.use_moe,
                "use_rag": self.rag.use_rag
            },
            "training": {
                "epochs": self.training.num_train_epochs,
                "batch_size": self.training.per_device_train_batch_size,
                "learning_rate": self.training.learning_rate,
                "use_wandb": self.training.use_wandb
            },
            "features": {
                "feedback": self.feedback.collect_user_feedback,
                "memory": self.memory.use_semantic_memory,
                "personas": len(self.persona.available_personas)
            }
        }
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Atualiza configuração a partir de dicionário"""
        for section_name, section_updates in updates.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_updates.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                        logger.info(f"Configuração atualizada: {section_name}.{key} = {value}")
                    else:
                        logger.warning(f"Configuração desconhecida ignorada: {section_name}.{key}")
            else:
                logger.warning(f"Seção desconhecida ignorada: {section_name}")