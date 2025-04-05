import os
import platform
import logging
import torch
import gc
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    """Perfil de hardware detectado para adaptação dinâmica"""
    cpu_count: int
    ram_gb: float
    gpu_available: bool
    gpu_vram_gb: float = 0.0
    gpu_name: str = "N/A"
    has_cuda: bool = False
    has_mps: bool = False  # Metal Performance Shaders (Apple Silicon)
    system_type: str = "unknown"  # high-end, mid-range, low-end
    
    def __str__(self):
        """Representação legível do perfil de hardware"""
        device = f"GPU: {self.gpu_name} ({self.gpu_vram_gb:.1f}GB)" if self.gpu_available else "CPU"
        return (f"Hardware: {self.system_type} | {device} | "
                f"CPUs: {self.cpu_count} | RAM: {self.ram_gb:.1f}GB")
    
    def get_model_size_params(self):
        """Retorna configurações de tamanho de modelo apropriadas para o hardware"""
        # Parâmetros: n_layer, n_head, n_embd, n_positions
        if self.system_type == "high-end":
            return 12, 12, 768, 2048  # Maior modelo
        elif self.system_type == "mid-range":
            return 8, 8, 512, 1024  # Modelo médio
        else:
            # Para sistema low-end com pouca RAM, reduzir ainda mais
            if self.ram_gb < 4.0:
                return 2, 2, 128, 256  # Modelo ultra-leve
            return 4, 4, 256, 512  # Modelo pequeno
    
    def get_optimal_batch_size(self):
        """Calcula o tamanho de batch ideal baseado no hardware"""
        if self.gpu_available:
            # Batch size baseado em VRAM disponível
            if self.gpu_vram_gb >= 16:
                return 16
            elif self.gpu_vram_gb >= 8:
                return 8
            elif self.gpu_vram_gb >= 4:
                return 4
            else:
                return 2
        else:
            # Batch size baseado em RAM para CPU
            if self.ram_gb >= 32:
                return 4
            elif self.ram_gb >= 16:
                return 2
            else:
                return 1
    
    def get_fallback_device_setup(self):
        """Configurações específicas para dispositivos com pouca memória"""
        result = {
            "use_8bit_quantization": self.system_type == "low-end",
            "use_4bit_quantization": self.system_type == "low-end" and self.ram_gb < 8,
            "use_cpu_offloading": self.system_type == "low-end",
            "use_checkpoint_activation": True,  # Economizar memória em todos os casos
            "low_cpu_mem_usage": self.ram_gb < 16,
            "max_memory_mb": int(min(self.ram_gb * 1024 * 0.8, 
                                self.gpu_vram_gb * 1024 * 0.8 if self.gpu_available else float('inf')))
        }
        return result

def detect_hardware():
    """Detecta e retorna um perfil do hardware disponível"""
    try:
        # Importar psutil apenas se disponível
        try:
            import psutil
            has_psutil = True
        except ImportError:
            has_psutil = False
            logger.warning("psutil não encontrado. Informações de hardware limitadas.")
        
        # Detectar CPU
        cpu_count = os.cpu_count() or 1
        
        # Detectar memória RAM
        if has_psutil:
            ram_info = psutil.virtual_memory()
            ram_gb = ram_info.total / (1024 ** 3)  # Converter para GB
        else:
            # Estimativa conservadora se psutil não está disponível
            ram_gb = 8.0
        
        # Verificar GPU
        gpu_available = torch.cuda.is_available() or hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        gpu_vram_gb = 0.0
        gpu_name = "N/A"
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        if has_cuda:
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            try:
                # Tentativa de obter a VRAM em alguns sistemas
                gpu_props = torch.cuda.get_device_properties(current_device)
                gpu_vram_gb = gpu_props.total_memory / (1024**3)
            except:
                # Estimativa aproximada se não conseguir obter diretamente
                gpu_vram_gb = 4.0  # Estimativa conservadora
        elif has_mps:
            gpu_name = "Apple Silicon GPU"
            # Estimativa aproximada para Apple Silicon (compartilha a RAM)
            gpu_vram_gb = ram_gb / 2
        
        # Classificar o sistema
        if gpu_available and (gpu_vram_gb >= 8 or has_mps) and ram_gb >= 16:
            system_type = "high-end"
        elif (gpu_available and gpu_vram_gb >= 4) or ram_gb >= 8:
            system_type = "mid-range"
        else:
            system_type = "low-end"
        
        profile = HardwareProfile(
            cpu_count=cpu_count,
            ram_gb=ram_gb,
            gpu_available=gpu_available,
            gpu_vram_gb=gpu_vram_gb,
            gpu_name=gpu_name,
            has_cuda=has_cuda,
            has_mps=has_mps,
            system_type=system_type
        )
        
        logger.info(f"Hardware detectado: {profile}")
        return profile
        
    except Exception as e:
        logger.error(f"Erro ao detectar hardware: {e}")
        # Retornar perfil mínimo seguro em caso de erro
        return HardwareProfile(
            cpu_count=1,
            ram_gb=4.0,
            gpu_available=False,
            system_type="low-end"
        )

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
    if hardware_profile.system_type == "low-end":
        config.num_hidden_layers = min(config.num_hidden_layers, 4)
        config.hidden_size = min(config.hidden_size, 256)
        config.num_attention_heads = min(config.num_attention_heads, 4)
    elif hardware_profile.system_type == "high-end":
        config.num_hidden_layers = max(config.num_hidden_layers, 24)
        config.hidden_size = max(config.hidden_size, 1024)
        config.num_attention_heads = max(config.num_attention_heads, 16)
    return config