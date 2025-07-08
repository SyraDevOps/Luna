import gc
import os
import sys
import time
import logging
import psutil
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class HardwareProfile:
    def __init__(self, cpu_count, ram_gb):
        self.cpu_count = cpu_count
        self.ram_gb = ram_gb
        self.gpu_available = False  # Valor padrão
        self.gpu_memory_gb = 0     # Valor padrão
        self.system_type = self._determine_system_type()

    def _determine_system_type(self):
        """Determina o tipo de sistema com base no hardware detectado."""
        if self.ram_gb < 8 or self.cpu_count <= 4:
            return "low-end"
        elif 8 <= self.ram_gb < 16 or 4 < self.cpu_count <= 8:
            return "mid-range"
        else:
            return "high-end"

def detect_hardware():
    """Detecta o hardware disponível."""
    
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)  # RAM em GB
    
    hardware_profile = HardwareProfile(cpu_count=cpu_count, ram_gb=ram_gb)
    
    # Detectar GPU
    hardware_profile.gpu_available = torch.cuda.is_available()
    if hardware_profile.gpu_available:
        try:
            # Obter memória total da GPU (em GB)
            gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            hardware_profile.gpu_memory_gb = round(gpu_memory_bytes / (1024 ** 3), 1)
        except Exception as e:
            logger.warning(f"Falha ao obter informações da GPU: {e}")
            hardware_profile.gpu_available = False
    
    return hardware_profile

def setup_memory_efficient_training():
    """Configurações para treinamento eficiente em memória"""
    try:
        # Limpar caches de memória
        gc.collect()
        
        # Limpar caches de CUDA se disponível
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Configurar PyTorch para usar TF32 se disponível (melhor precisão/performance)
        if torch.cuda.is_available():
            # Safe versioning check
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
        
        # Coletar lixo novamente após configuração
        gc.collect()
        
    except Exception as e:
        logger.error(f"Erro ao configurar memória eficiente: {e}")

def adjust_model_for_hardware(config, hardware_profile):
    """Ajusta os parâmetros do modelo com base no hardware."""
    if hardware_profile.ram_gb < 8:
        config.num_hidden_layers = min(config.num_hidden_layers, 4)
        config.hidden_size = min(config.hidden_size, 256)
        config.num_attention_heads = min(config.num_attention_heads, 4)
    elif hardware_profile.ram_gb >= 16:
        config.num_hidden_layers = max(config.num_hidden_layers, 24)
        config.hidden_size = max(config.hidden_size, 1024)
        config.num_attention_heads = max(config.num_attention_heads, 16)
    return config

def adjust_model_config_dynamically(config, hardware_profile):
    """Ajusta dinamicamente a configuração do modelo baseado no hardware"""
    if hardware_profile.system_type == "low-end":
        config.num_hidden_layers = 4
        config.hidden_size = 256
        config.num_attention_heads = 4
    elif hardware_profile.system_type == "mid-range":
        config.num_hidden_layers = 8
        config.hidden_size = 512
        config.num_attention_heads = 8
    else:  # high-end
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.num_attention_heads = 12
    
    return config