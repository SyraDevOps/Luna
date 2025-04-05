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
        stanza_dir = os.path.join(tempfile.gettempdir(), "stanza_resources")
        os.makedirs(stanza_dir, exist_ok=True)
        os.environ["STANZA_RESOURCES_DIR"] = stanza_dir
        stanza.download('pt')
        result["stanza_available"] = True
    except ImportError:
        logger.warning("Stanza não está disponível. Algumas funcionalidades podem estar limitadas.")
    except Exception as e:
        logger.warning(f"Erro ao carregar Stanza: {e}")
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
        
        # Expor atributos importantes diretamente na classe para facilitar testes
        self.attention_heads = getattr(config, 'num_attention_heads', 12)
        self.hidden_size = getattr(config, 'hidden_size', 768)
        self.num_layers = getattr(config, 'num_hidden_layers', 12)
        
        # Ajustar com base no hardware
        if self.hardware_profile.system_type == "low-end":
            # Verificar se o atributo existe antes de ajustar
            if hasattr(self.config, 'num_attention_heads'):
                old_heads = self.config.num_attention_heads
                self.config.num_attention_heads = min(self.config.num_attention_heads, 2)
                self.attention_heads = self.config.num_attention_heads  # Atualizar o atributo exposto
                if old_heads != self.config.num_attention_heads:
                    self.logger.warning(f"Reduzindo cabeças de atenção de {old_heads} para {self.config.num_attention_heads} baseado no hardware")
            
            # Ajustar outros parâmetros para hardware de baixo desempenho
            if hasattr(self.config, 'hidden_size') and self.config.hidden_size > 256:
                old_size = self.config.hidden_size
                self.config.hidden_size = 256
                self.hidden_size = self.config.hidden_size  # Atualizar o atributo exposto
                self.logger.warning(f"Reduzindo tamanho oculto de {old_size} para {self.config.hidden_size} baseado no hardware")
    
    @classmethod
    def from_scratch(cls, config, use_lightweight=False):
        """Cria um modelo do zero com base na configuração fornecida."""
        try:
            hardware_profile = detect_hardware()
            if use_lightweight and hardware_profile.system_type == "low-end":
                config.hidden_size = min(config.hidden_size, 128)
                config.num_attention_heads = min(config.num_attention_heads, 2)
                config.num_hidden_layers = min(config.num_hidden_layers, 4)
                logging.warning("Configuração ajustada para hardware de baixo desempenho.")
            
            # Criar instância da classe
            instance = cls(config)
            
            # Mapear atributos de nossa config para a GPT2Config
            gpt2_config_kwargs = {
                'vocab_size': getattr(config, 'vocab_size', 50257),
                'n_embd': getattr(config, 'hidden_size', 768),
                'n_layer': getattr(config, 'num_hidden_layers', 12),
                'n_head': getattr(config, 'num_attention_heads', 12),
                'n_positions': getattr(config, 'n_positions', 1024),
                'n_ctx': getattr(config, 'n_ctx', 1024),
                'activation_function': getattr(config, 'activation_function', 'gelu'),
                'resid_pdrop': getattr(config, 'dropout_rate', 0.1),
                'embd_pdrop': getattr(config, 'dropout_rate', 0.1),
                'attn_pdrop': getattr(config, 'dropout_rate', 0.1),
                'scale_attn_weights': True
            }
            
            # Criar objeto GPT2Config
            from transformers import GPT2Config
            gpt2_config = GPT2Config(**gpt2_config_kwargs)
            
            # Inicializar o modelo GPT-2 com a configuração adequada
            from transformers import GPT2LMHeadModel
            instance.model = GPT2LMHeadModel(config=gpt2_config)
            
            # Não vamos usar nosso método manual de inicialização que está causando problemas
            # O modelo GPT2 já vem com inicialização própria
            # instance.init_weights()
            
            logging.info("Modelo criado do zero com sucesso.")
            return instance
        except Exception as e:
            logging.error(f"Erro ao criar modelo: {str(e)}")
            raise
    
    @classmethod
    def from_pretrained(cls, model_dir, config=None, device_setup=None):
        """Carrega um modelo pré-treinado com suporte para adaptação ao hardware."""
        try:
            logger = logging.getLogger(__name__)
            
            # Se não foram fornecidas configurações de dispositivo, criar padrões
            if device_setup is None:
                device_setup = {
                    'low_cpu_mem_usage': True,
                    'use_fp16': False
                }
            
            # Se não foi fornecida configuração, tentar carregar do diretório
            if config is None:
                config_path = os.path.join(model_dir, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_dict = json.load(f)
                    
                    # Criar objeto de configuração
                    from src.config.config import ModelConfig
                    config = ModelConfig()
                    
                    # Copiar atributos do dicionário para o objeto de configuração
                    for key, value in config_dict.items():
                        setattr(config, key, value)
                    
                    logger.info(f"Configuração carregada de {config_path}")
                else:
                    from src.config.config import Config
                    logger.warning(f"Arquivo de configuração não encontrado em {config_path}. Usando configuração padrão.")
                    config = Config().model
            
            # Criar instância do modelo com a configuração
            instance = cls(config)
            
            # Carregar modelo usando transformers
            from transformers import GPT2LMHeadModel
            instance.model = GPT2LMHeadModel.from_pretrained(
                model_dir,
                low_cpu_mem_usage=device_setup.get('low_cpu_mem_usage', True)
            )
            
            # Expor atributos importantes para testes
            instance.attention_heads = getattr(config, 'num_attention_heads', 
                                              getattr(instance.model.config, 'n_head', 12))
            instance.hidden_size = getattr(config, 'hidden_size', 
                                          getattr(instance.model.config, 'n_embd', 768))
            
            logger.info(f"Modelo carregado com sucesso de {model_dir}")
            return instance
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    @classmethod
    def load_from_wandb(cls, run_path, config=None):
        """Carrega um modelo diretamente do Weights & Biases.
        
        Args:
            run_path: Caminho do run no formato "entity/project/run_id"
            config: Configuração do modelo a ser utilizada
            
        Returns:
            Uma instância do modelo carregada do wandb
        """
        try:
            import wandb
            
            # Fazer download do modelo
            api = wandb.Api()
            run = api.run(run_path)
            model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
            
            if not model_artifacts:
                raise ValueError(f"Nenhum artefato de modelo encontrado no run {run_path}")
            
            # Pegar o artefato mais recente
            model_artifact = model_artifacts[-1]
            model_dir = model_artifact.download()
            
            # Carregar o modelo usando o método regular
            instance = cls.from_pretrained(model_dir, config)
            logger.info(f"Modelo carregado com sucesso do wandb: {run_path}")
            return instance
        except ImportError:
            logger.error("Weights & Biases (wandb) não está instalado. Use pip install wandb para instalá-lo.")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar modelo do wandb: {str(e)}")
            raise
    
    def to_appropriate_device(self, force_device=None):
        """Move o modelo para o melhor dispositivo disponível ou especificado.
        
        Args:
            force_device: Se fornecido, força o uso deste dispositivo ('cpu', 'cuda', 'mps')
        """
        if not self.model:
            logger.warning("Modelo não inicializado, não há nada para mover para o dispositivo")
            return
        
        # Determinar o melhor dispositivo
        if force_device:
            if force_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA solicitado, mas não disponível. Usando CPU.")
                device = torch.device("cpu")
            elif force_device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS (Apple Silicon) solicitado, mas não disponível. Usando CPU.")
                device = torch.device("cpu")
            else:
                device = torch.device(force_device)
                logger.info(f"Usando dispositivo forçado: {force_device}")
        else:
            # Verificar as capacidades de hardware de forma segura
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            
            # Propriedades do hardware podem estar em diferentes formatos dependendo da versão
            # Por isso usamos uma abordagem defensiva
            if has_cuda:
                device = torch.device("cuda")
                gpu_name = getattr(self.hardware_profile, 'gpu_name', 'GPU')
                logger.info(f"Movendo modelo para GPU: {gpu_name}")
            elif has_mps:
                device = torch.device("mps")
                logger.info("Movendo modelo para Apple Silicon GPU")
            else:
                device = torch.device("cpu")
                logger.info("Usando CPU para o modelo")
        
        # Mover modelo para o dispositivo com tratamento de erros de memória
        try:
            self.model.to(device)
            return device
        except RuntimeError as e:
            if "out of memory" in str(e) and device.type != "cpu":
                logger.warning(f"{device.type.upper()} sem memória suficiente, usando CPU como fallback")
                self.model.to(torch.device("cpu"))
                return torch.device("cpu")
            else:
                raise
    
    def save(self, model_dir, save_to_wandb=False, run_name=None):
        """Salva o modelo no diretório especificado e opcionalmente no Weights & Biases."""
        try:
            if self.model is None:
                raise ValueError("O modelo não foi inicializado. Não é possível salvar.")
            
            # Criar diretório se não existir
            os.makedirs(model_dir, exist_ok=True)
            
            # Salvar configuração
            config_path = os.path.join(model_dir, "config.json")
            config_dict = {k: v for k, v in self.config.__dict__.items() 
                          if not k.startswith('_') and not callable(v)}
            
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=4)
            self.logger.info(f"Configuração salva em {config_path}")
            
            # Limpar memória antes de salvar para evitar problemas de bloqueio de arquivo
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Salvar modelo em um diretório temporário primeiro para evitar conflitos
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Salvar modelo direto com o método do transformers no diretório temporário
                    temp_save_path = os.path.join(temp_dir, "temp_model")
                    self.model.save_pretrained(temp_save_path, safe_serialization=True)
                    
                    # Copiar os arquivos para o destino final
                    import shutil
                    
                    # Verificar se o diretório de destino contém arquivos que poderiam causar conflito
                    safe_files = [f for f in os.listdir(model_dir) if not f.endswith('.bin') and not f.endswith('.safetensors')]
                    for item in os.listdir(temp_save_path):
                        source = os.path.join(temp_save_path, item)
                        destination = os.path.join(model_dir, item)
                        
                        # Se for um arquivo existente que poderia causar bloqueio, remover primeiro
                        if os.path.exists(destination):
                            try:
                                if os.path.isfile(destination):
                                    os.unlink(destination)
                                elif os.path.isdir(destination):
                                    shutil.rmtree(destination)
                            except Exception as e:
                                self.logger.warning(f"Não foi possível remover arquivo existente {destination}: {str(e)}")
                        
                        # Copiar arquivo do temporário para o destino final
                        if os.path.isfile(source):
                            shutil.copy2(source, destination)
                        else:
                            shutil.copytree(source, destination)
                            
                    self.logger.info(f"Modelo salvo com sucesso em {model_dir}")
                except Exception as e:
                    self.logger.error(f"Erro ao salvar modelo: {str(e)}")
                    # Segunda tentativa usando apenas arquivos fundamentais
                    try:
                        # Salvar apenas usando state_dict
                        torch.save(self.model.state_dict(), os.path.join(temp_dir, "pytorch_model.bin"))
                        shutil.copy2(os.path.join(temp_dir, "pytorch_model.bin"), os.path.join(model_dir, "pytorch_model.bin"))
                        self.logger.info(f"Modelo salvo com método alternativo em {model_dir}")
                    except Exception as e2:
                        self.logger.error(f"Falha também no método alternativo: {str(e2)}")
                        raise
            
            # Salvar no wandb se solicitado
            if save_to_wandb:
                self._save_to_wandb(model_dir, run_name)
                
            return True
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar o modelo: {str(e)}")
            raise

    def _save_to_wandb(self, model_dir, run_name=None):
        """Salva o modelo no Weights & Biases."""
        try:
            import wandb
            
            # Configurar wandb se não estiver já configurado
            if wandb.run is None:
                if run_name is None:
                    run_name = os.path.basename(model_dir)
                wandb.init(project="lunagpt", name=run_name)
            
            # Criar artefato e fazer upload
            artifact = wandb.Artifact(f"{run_name}_model", type="model")
            artifact.add_dir(model_dir)
            wandb.log_artifact(artifact)
            self.logger.info(f"Modelo salvo com sucesso no wandb como {run_name}_model")
            
        except ImportError:
            self.logger.warning("Weights & Biases (wandb) não está instalado. Instalando agora...")
            try:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb>=0.15.0"])
                self.logger.info("wandb instalado com sucesso. Para salvar no wandb, execute novamente.")
            except Exception as e:
                self.logger.error(f"Erro ao instalar wandb: {str(e)}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo no wandb: {str(e)}")

    def init_weights(self):
        """Inicializa os pesos do modelo."""
        if self.model:
            self.model.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        """Inicializa pesos de um módulo com configuração adequada para cada tipo de camada."""
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) >= 2:  # Para tensores 2D ou maior
                torch.nn.init.xavier_uniform_(module.weight)
            else:  # Para tensores 1D como bias ou embedding
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
        # Inicialização de bias
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def generate(self, input_text, tokenizer, max_length=100, temperature=0.8, top_p=0.95, repetition_penalty=1.2):
        """Gera texto a partir de um prompt de entrada.
        
        Args:
            input_text: Texto de entrada para o modelo
            tokenizer: Tokenizer para processar o texto
            max_length: Comprimento máximo da sequência gerada
            temperature: Controle de aleatoriedade (menor = mais determinístico)
            top_p: Probabilidade cumulativa para amostragem de nucleus
            repetition_penalty: Penalidade para tokens repetidos
            
        Returns:
            Texto gerado pelo modelo
        """
        if self.model is None:
            raise ValueError("Modelo não inicializado. Não é possível gerar texto.")
        
        # Tokenizar entrada
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
        # Verificar dispositivo do modelo e mover input_ids
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Gerar saída
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decodificar e retornar
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

def advanced_augment(text: str) -> str:
    # Implementação da função
    return text

def syntactic_reorder(text: str) -> str:
    # Implementação da função
    return text