import os
import platform
import logging
import torch
import gc
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class HardwareProfile:
    def __init__(self, cpu_count, ram_gb):
        self.cpu_count = cpu_count
        self.ram_gb = ram_gb
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
    import psutil
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)  # RAM em GB
    return HardwareProfile(cpu_count=cpu_count, ram_gb=ram_gb)

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
            if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
        
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