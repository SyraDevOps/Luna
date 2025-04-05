import os
import json
import torch
import torch.nn as nn
import logging
import gc
import time
import tempfile
from pathlib import Path
import numpy as np
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast, AutoModelForCausalLM, AutoConfig
from typing import Dict, Any, Optional
from src.config.config import Config  # Certifique-se de importar Config corretamente
from src.utils.logging_utils import logger
from src.config.config import ModelConfig
from src.models.tokenizer import LunaTokenizer  # Certifique-se de importar LunaTokenizer corretamente
from src.utils.hardware_utils import detect_hardware, HardwareProfile, setup_memory_efficient_training

logger = logging.getLogger(__name__)

def load_optional_dependencies():
    """Carrega dependências opcionais como stanza."""
    result = {"stanza_available": False}
    try:
        import stanza
        # Use um diretório temporário específico para o download
        stanza_dir = os.path.join(tempfile.gettempdir(), "stanza_resources")
        os.makedirs(stanza_dir, exist_ok=True)
        
        try:
            # Definir diretório via variável de ambiente e então fazer o download
            os.environ["STANZA_RESOURCES_DIR"] = stanza_dir
            stanza.download('pt')  # Remova o parâmetro 'dir' que não é suportado
            result["stanza_available"] = True
        except (PermissionError, OSError) as e:
            logger.warning(f"Não foi possível baixar recursos do Stanza. Erro: {e}")
            logger.warning("Continuando sem recursos do Stanza. Algumas funcionalidades podem estar limitadas.")
    except ImportError:
        logger.warning("Stanza não está disponível. Algumas funcionalidades de processamento de linguagem podem estar limitadas.")
    
    return result

# Tentativa de carregar dependências opcionais, mas não falhar se não puder
try:
    OPTIONAL = load_optional_dependencies()
except Exception as e:
    logger.warning(f"Erro ao carregar dependências opcionais: {e}")
    OPTIONAL = {"stanza_available": False}

__all__ = ["OPTIONAL", "advanced_augment", "syntactic_reorder"]

class StateSpaceLayer(nn.Module):
    """Camada StateSpace simples para adaptação dinâmica"""
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

class MoEBlock(nn.Module):
    """Bloco de Mixture of Experts para melhor especialização"""
    def __init__(self, input_dim: int, num_experts: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(input_dim, num_experts)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calcular pesos de roteamento
        routing_weights = torch.softmax(self.router(x), dim=-1)
        
        # Aplicar cada especialista
        expert_outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            expert_outputs += routing_weights[..., i:i+1] * expert_output
            
        return expert_outputs

class LunaModel:
    """Modelo neural de base para o sistema Luna"""
    
    def __init__(self, config, hardware_profile=None):
        """Inicializa o modelo Luna."""
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.hardware_profile = hardware_profile or detect_hardware()
        
        # Definir parâmetros padrão
        self.attention_heads = 12
        
        # Ajustar com base no hardware
        # Converta para string para verificação segura
        hardware_str = str(self.hardware_profile)
        if "low-end" in hardware_str:
            self.attention_heads = 2
            self.logger.warning(f"Reduzindo cabeças de atenção de 12 para {self.attention_heads} baseado no hardware")
    
    @classmethod
    def from_scratch(cls, config):
        """Cria um novo modelo do zero com a configuração especificada."""
        try:
            # Detectar hardware e ajustar parâmetros
            hardware_profile = detect_hardware()
            
            # Verificar se a otimização para hardware leve está ativada (opcional)
            use_lightweight = getattr(config, 'use_lightweight_mode', True)
            
            # Redução para hardware leve (apenas se a opção estiver ativada)
            if use_lightweight and hardware_profile.system_type == "low-end":
                original_hidden_size = config.hidden_size
                original_heads = config.num_attention_heads
                
                # Reduzir dimensões para hardware leve
                config.hidden_size = min(config.hidden_size, 128)
                config.num_attention_heads = min(config.num_attention_heads, 2)
                
                logger.warning(f"Reduzindo hidden_size de {original_hidden_size} para {config.hidden_size} baseado no hardware")
                logger.warning(f"Reduzindo cabeças de atenção de {original_heads} para {config.num_attention_heads} baseado no hardware")
            
            # Criar instância e modelo
            instance = cls(config)
            
            # Definir n_positions se não existir na configuração
            n_positions = getattr(config, 'n_positions', 1024)
            
            model_config = GPT2Config(
                vocab_size=config.vocab_size,
                n_positions=n_positions,
                n_ctx=n_positions,
                n_embd=config.hidden_size,
                n_layer=config.num_hidden_layers,
                n_head=config.num_attention_heads,
            )
            instance.model = GPT2LMHeadModel(model_config)
            
            logger.info("Modelo criado do zero com sucesso.")
            return instance
        except Exception as e:
            logger.error(f"Erro ao criar modelo: {str(e)}")
            raise
    
    @classmethod
    def from_pretrained(cls, model_dir, config=None, device_setup=None):
        """Carrega um modelo pré-treinado com suporte para adaptação ao hardware."""
        try:
            # Se não foram fornecidas configurações, criar padrões
            if device_setup is None:
                device_setup = {
                    'low_cpu_mem_usage': True,
                    'use_fp16': False
                }
            
            # Verificar dimensões salvas no checkpoint para garantir compatibilidade
            checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
            
            # Carregar configuração salva se existir
            config_path = os.path.join(model_dir, "config.json")
            vocab_size = None
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    vocab_size = saved_config.get('vocab_size', None)
                    
            # Examinar checkpoint para dimensões
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                
                # Verificar dimensões dos embeddings
                if "transformer.wte.weight" in state_dict:
                    embedding_shape = state_dict["transformer.wte.weight"].shape
                    embedding_size = embedding_shape[1]  # segunda dimensão é hidden_size
                    vocab_size = embedding_shape[0] if vocab_size is None else vocab_size
                    
                    logger.info(f"Detectado embedding_size={embedding_size} no checkpoint")
                    logger.info(f"Detectado vocab_size={vocab_size} no checkpoint")
                    
                    # Ajustar config para corresponder ao checkpoint
                    if config is not None:
                        if hasattr(config, 'hidden_size'):
                            original_size = config.hidden_size
                            config.hidden_size = embedding_size
                            logger.info(f"Ajustando hidden_size de {original_size} para {embedding_size}")
                        
                        if hasattr(config, 'vocab_size'):
                            original_vocab = config.vocab_size
                            config.vocab_size = vocab_size
                            logger.info(f"Ajustando vocab_size de {original_vocab} para {vocab_size}")
            
            # Criar instância e configurar modelo
            instance = cls(config)
            
            # Configurar parâmetros do modelo para corresponder ao checkpoint
            config_dict = {'revision': 'main'}
            if hasattr(config, 'hidden_size'):
                config_dict['hidden_size'] = config.hidden_size
            if hasattr(config, 'num_attention_heads'):
                config_dict['num_attention_heads'] = config.num_attention_heads
            if vocab_size is not None:
                config_dict['vocab_size'] = vocab_size
            
            # Carregar modelo com configuração compatível
            instance.model = GPT2LMHeadModel.from_pretrained(
                model_dir,
                low_cpu_mem_usage=device_setup['low_cpu_mem_usage'],
                torch_dtype=torch.float16 if device_setup.get('use_fp16', False) else torch.float32,
                **config_dict
            )
            
            logger.info(f"Modelo carregado com sucesso de {model_dir}")
            return instance
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    def to_appropriate_device(self):
        """Move o modelo para o melhor dispositivo disponível"""
        if not self.model:
            return
        
        # Determinar o melhor dispositivo
        if self.hardware_profile.has_cuda:
            device = torch.device("cuda")
            logger.info(f"Movendo modelo para GPU: {self.hardware_profile.gpu_name}")
        elif self.hardware_profile.has_mps:
            device = torch.device("mps")
            logger.info("Movendo modelo para Apple Silicon GPU")
        else:
            device = torch.device("cpu")
            logger.info("Usando CPU para o modelo")
        
        # Mover modelo para o dispositivo
        try:
            self.model.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("GPU sem memória suficiente, usando CPU como fallback")
                self.model.to(torch.device("cpu"))
            else:
                raise
    
    def save(self, model_dir):
        """Salva o modelo no diretório especificado."""
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Salvar configuração completa junto com o modelo
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'w') as f:
                # Obter dimensões atuais do modelo
                config_dict = {
                    'hidden_size': self.model.config.hidden_size,
                    'num_attention_heads': self.model.config.num_attention_heads,
                    'vocab_size': self.model.config.vocab_size,
                    'n_positions': self.model.config.n_positions,
                    'n_layer': self.model.config.n_layer
                }
                json.dump(config_dict, f)
                logger.info(f"Configuração salva em {config_path}")
            
            # Salvar pesos do modelo
            model_path = os.path.join(model_dir, "pytorch_model.bin")
            torch.save(self.model.state_dict(), model_path)
            
            logger.info(f"Modelo salvo com sucesso em: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar o modelo: {str(e)}")
            raise

def advanced_augment(text: str) -> str:
    # Implementação da função
    return text

def syntactic_reorder(text: str) -> str:
    # Implementação da função
    return text