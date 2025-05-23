import os
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
from transformers import (
    PreTrainedTokenizerFast, 
    GPT2LMHeadModel, 
    GPT2Config, 
    GPT2TokenizerFast, 
    AutoModelForCausalLM, 
    AutoConfig
)

from src.config.config import ModelConfig
from src.models.moe import MoEBlock
from src.models.hypernet import HyperNetwork
from src.models.growing_network import GrowingNetwork
from src.utils.hardware_utils import detect_hardware

logger = logging.getLogger(__name__)

class LunaModel(nn.Module):
    """Modelo principal do LunaGPT com arquitetura híbrida"""
    
    def __init__(self, config: ModelConfig):
        """
        Inicializa o modelo Luna
        
        Args:
            config: Configuração do modelo
        """
        super().__init__()
        self.config = config
        self.device_info = detect_hardware()
        
        # Ajustar configuração baseada no hardware
        self._adjust_config_for_hardware()
        
        # Criar configuração do transformers
        self.transformer_config = self._create_transformer_config()
        
        # Modelo base (GPT-2 como backbone)
        self.model = GPT2LMHeadModel(self.transformer_config)
        
        # Componentes especializados
        self.moe_blocks = None
        self.hypernet = None
        self.growing_network = None
        
        # Inicializar componentes opcionais
        self._initialize_optional_components()
        
        # Mover para dispositivo apropriado
        self.to_appropriate_device()
        
        logger.info(f"Modelo Luna inicializado com {self.get_parameter_count():,} parâmetros")
    
    def _adjust_config_for_hardware(self):
        """Ajusta configuração baseada no hardware disponível"""
        if self.device_info.system_type == "low-end":
            # Reduzir tamanho do modelo para hardware limitado
            self.config.hidden_size = min(self.config.hidden_size, 512)
            self.config.num_hidden_layers = min(self.config.num_hidden_layers, 6)
            self.config.num_attention_heads = min(self.config.num_attention_heads, 8)
            self.config.intermediate_size = min(self.config.intermediate_size, 2048)
            
            # Desabilitar componentes pesados
            self.config.use_moe = False
            self.config.use_hypernet = False
            self.config.use_growing_network = False
            
            logger.info("Configuração ajustada para hardware de baixo desempenho")
        
        elif self.device_info.system_type == "mid-range":
            # Configuração moderada
            self.config.hidden_size = min(self.config.hidden_size, 768)
            self.config.num_hidden_layers = min(self.config.num_hidden_layers, 8)
            
            logger.info("Configuração ajustada para hardware de médio desempenho")
    
    def _create_transformer_config(self) -> GPT2Config:
        """Cria configuração do transformers"""
        return GPT2Config(
            vocab_size=self.config.vocab_size,
            n_positions=self.config.max_position_embeddings,
            n_embd=self.config.hidden_size,
            n_layer=self.config.num_hidden_layers,
            n_head=self.config.num_attention_heads,
            n_inner=self.config.intermediate_size,
            resid_pdrop=self.config.dropout_rate,
            attn_pdrop=self.config.attention_dropout,
            layer_norm_epsilon=self.config.layer_norm_eps,
            initializer_range=self.config.initializer_range,
            use_cache=self.config.use_cache,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    
    def _initialize_optional_components(self):
        """Inicializa componentes opcionais baseados na configuração"""
        try:
            # Mixture of Experts
            if self.config.use_moe:
                self.moe_blocks = nn.ModuleList([
                    MoEBlock(
                        hidden_size=self.config.hidden_size,
                        num_experts=self.config.num_experts,
                        top_k=self.config.top_k_experts
                    ) for _ in range(self.config.num_hidden_layers // 2)
                ])
                logger.info(f"Inicializados {len(self.moe_blocks)} blocos MoE")
            
            # HyperNetwork
            if self.config.use_hypernet:
                self.hypernet = HyperNetwork(
                    input_dim=self.config.hidden_size,
                    hidden_dim=self.config.hidden_size // 2,
                    output_dim=self.config.hidden_size
                )
                logger.info("HyperNetwork inicializada")
            
            # Growing Network
            if self.config.use_growing_network:
                self.growing_network = GrowingNetwork(
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_hidden_layers
                )
                logger.info("Growing Network inicializada")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar componentes opcionais: {e}")
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass do modelo"""
        # Forward pass no modelo base
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # Aplicar componentes especializados se disponíveis
        if self.moe_blocks is not None:
            hidden_states = outputs.last_hidden_state
            
            # Aplicar MoE em camadas alternadas
            for i, moe_block in enumerate(self.moe_blocks):
                hidden_states = moe_block(hidden_states)
            
            # Atualizar outputs
            outputs.last_hidden_state = hidden_states
        
        # Aplicar HyperNetwork se disponível
        if self.hypernet is not None:
            hidden_states = outputs.last_hidden_state
            hyper_weights = self.hypernet(hidden_states.mean(dim=1))  # Global pooling
            hidden_states = hidden_states * hyper_weights.unsqueeze(1)
            outputs.last_hidden_state = hidden_states
        
        # Aplicar Growing Network se disponível
        if self.growing_network is not None:
            hidden_states = outputs.last_hidden_state
            hidden_states = self.growing_network(hidden_states)
            outputs.last_hidden_state = hidden_states
        
        return outputs
    
    def generate(self, input_ids, **kwargs):
        """Gera texto usando o modelo"""
        return self.model.generate(input_ids, **kwargs)
    
    def to_appropriate_device(self):
        """Move modelo para dispositivo apropriado"""
        if torch.cuda.is_available() and self.device_info.gpu_available:
            device = torch.cuda.current_device()
            self.to(device)
            logger.info(f"Modelo movido para GPU: {torch.cuda.get_device_name(device)}")
        else:
            self.to('cpu')
            logger.info("Modelo executando em CPU")
    
    def get_parameter_count(self) -> int:
        """Retorna número total de parâmetros"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameter_count(self) -> int:
        """Retorna número de parâmetros treináveis"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, save_directory: str):
        """
        Salva o modelo completo
        
        Args:
            save_directory: Diretório onde salvar o modelo
        """
        try:
            os.makedirs(save_directory, exist_ok=True)
            
            # Salvar modelo base
            self.model.save_pretrained(save_directory)
            
            # Salvar configuração customizada
            config_path = os.path.join(save_directory, "luna_config.json")
            import json
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            # Salvar componentes especializados
            components_dir = os.path.join(save_directory, "components")
            os.makedirs(components_dir, exist_ok=True)
            
            if self.moe_blocks is not None:
                torch.save(self.moe_blocks.state_dict(), 
                          os.path.join(components_dir, "moe_blocks.pt"))
            
            if self.hypernet is not None:
                torch.save(self.hypernet.state_dict(), 
                          os.path.join(components_dir, "hypernet.pt"))
            
            if self.growing_network is not None:
                torch.save(self.growing_network.state_dict(), 
                          os.path.join(components_dir, "growing_network.pt"))
            
            # Salvar modelo completo como backup
            full_model_path = os.path.join(save_directory, "full_model.pt")
            torch.save(self, full_model_path)
            
            logger.info(f"Modelo salvo em {save_directory}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            raise
    
    @classmethod
    def from_pretrained(cls, model_directory: str, config: ModelConfig):
        """
        Carrega modelo de diretório
        
        Args:
            model_directory: Diretório do modelo
            config: Configuração do modelo
            
        Returns:
            Instância do LunaModel carregada
        """
        try:
            # Verificar se existe modelo completo salvo
            full_model_path = os.path.join(model_directory, "full_model.pt")
            if os.path.exists(full_model_path):
                logger.info(f"Carregando modelo completo de {full_model_path}")
                model = torch.load(full_model_path, map_location='cpu', weights_only=False)
                model.to_appropriate_device()
                return model
            
            # Carregar configuração customizada se existir
            config_path = os.path.join(model_directory, "luna_config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                
                # Atualizar configuração
                for key, value in saved_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Criar nova instância
            model = cls(config)
            
            # Carregar modelo base se existir
            if os.path.exists(os.path.join(model_directory, "pytorch_model.bin")):
                model.model = GPT2LMHeadModel.from_pretrained(model_directory)
                logger.info("Modelo base carregado")
            
            # Carregar componentes especializados
            components_dir = os.path.join(model_directory, "components")
            if os.path.exists(components_dir):
                
                # Carregar MoE
                moe_path = os.path.join(components_dir, "moe_blocks.pt")
                if os.path.exists(moe_path) and model.moe_blocks is not None:
                    model.moe_blocks.load_state_dict(torch.load(moe_path, map_location='cpu'))
                    logger.info("Blocos MoE carregados")
                
                # Carregar HyperNetwork
                hypernet_path = os.path.join(components_dir, "hypernet.pt")
                if os.path.exists(hypernet_path) and model.hypernet is not None:
                    model.hypernet.load_state_dict(torch.load(hypernet_path, map_location='cpu'))
                    logger.info("HyperNetwork carregada")
                
                # Carregar Growing Network
                growing_path = os.path.join(components_dir, "growing_network.pt")
                if os.path.exists(growing_path) and model.growing_network is not None:
                    model.growing_network.load_state_dict(torch.load(growing_path, map_location='cpu'))
                    logger.info("Growing Network carregada")
            
            model.to_appropriate_device()
            logger.info(f"Modelo carregado de {model_directory}")
            return model
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de {model_directory}: {e}")
            raise
    
    @classmethod
    def from_scratch(cls, config: ModelConfig):
        """
        Cria novo modelo do zero
        
        Args:
            config: Configuração do modelo
            
        Returns:
            Nova instância do LunaModel
        """
        logger.info("Criando novo modelo Luna do zero")
        return cls(config)
    
    def resize_token_embeddings(self, new_size: int):
        """
        Redimensiona embeddings de tokens
        
        Args:
            new_size: Novo tamanho do vocabulário
        """
        self.model.resize_token_embeddings(new_size)
        self.config.vocab_size = new_size
        logger.info(f"Vocabulário redimensionado para {new_size} tokens")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Retorna uso de memória do modelo"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return {
                "allocated_gb": memory_allocated,
                "cached_gb": memory_cached,
                "parameters": self.get_parameter_count(),
                "trainable_parameters": self.get_trainable_parameter_count()
            }
        else:
            return {
                "parameters": self.get_parameter_count(),
                "trainable_parameters": self.get_trainable_parameter_count()
            }
    
    def enable_gradient_checkpointing(self):
        """Habilita gradient checkpointing para economizar memória"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing habilitado")
    
    def disable_gradient_checkpointing(self):
        """Desabilita gradient checkpointing"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing desabilitado")
    
    def freeze_base_model(self):
        """Congela parâmetros do modelo base"""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Modelo base congelado")
    
    def unfreeze_base_model(self):
        """Descongela parâmetros do modelo base"""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Modelo base descongelado")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações detalhadas do modelo"""
        return {
            "model_name": self.config.model_name,
            "total_parameters": self.get_parameter_count(),
            "trainable_parameters": self.get_trainable_parameter_count(),
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "vocab_size": self.config.vocab_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "has_moe": self.moe_blocks is not None,
            "has_hypernet": self.hypernet is not None,
            "has_growing_network": self.growing_network is not None,
            "device": str(next(self.parameters()).device),
            "memory_usage": self.get_memory_usage()
        }