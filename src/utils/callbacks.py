import time
import logging
import os
import psutil
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CustomTrainingCallback(TrainerCallback):
    """Callback personalizado para monitorar treinamento"""
    
    def __init__(self, timeout_min=30, monitor_memory=False):
        """Inicializa o callback com um timeout em minutos"""
        self.timeout_min = timeout_min
        self.start_time = None
        self.timeout_time = None
        self.last_log_time = None
        self.log_interval = 60  # Logar a cada 60 segundos
        self.monitor_memory = monitor_memory
        self.peak_memory_usage = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Chamado quando o treinamento começa"""
        self.start_time = time.time()
        self.timeout_time = self.start_time + (self.timeout_min * 60)
        self.last_log_time = self.start_time
        
        logger.info(f"Iniciando treinamento com timeout de {self.timeout_min} minutos")
        
        # Log inicial de uso de recursos
        if self.monitor_memory:
            self._log_resource_usage()
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Chamado quando uma época começa"""
        logger.info(f"Iniciando época {state.epoch}/{args.num_train_epochs}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Chamado quando o trainer loga"""
        if not logs:
            return
        
        # Extrair métricas relevantes
        step = state.global_step
        epoch = state.epoch
        
        # Extrair e formatar métricas específicas
        metrics_str = []
        for key, value in logs.items():
            if key.startswith("train_") or key.startswith("eval_"):
                metrics_str.append(f"{key}={value:.4f}")
        
        if metrics_str:
            metrics_log = ", ".join(metrics_str)
            logger.info(f"Passo {step}, Época {epoch:.2f}: {metrics_log}")
            
            # Registrar uso de recursos periodicamente
            if self.monitor_memory:
                self._log_resource_usage()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Chamado após cada passo de treinamento"""
        # Verificar timeout
        current_time = time.time()
        
        # Verificar se devemos interromper o treinamento pelo timeout
        if current_time > self.timeout_time:
            logger.warning(f"Interrompendo treinamento após {self.timeout_min} minutos")
            control.should_training_stop = True
        
        # Logar progresso periodicamente
        if current_time - self.last_log_time >= self.log_interval:
            # Calcular tempo decorrido e tempo estimado restante
            elapsed_time = current_time - self.start_time
            progress = state.global_step / state.max_steps if state.max_steps > 0 else 0
            
            if progress > 0:
                estimated_total = elapsed_time / progress
                remaining = estimated_total - elapsed_time
                
                # Formatar tempos para exibição
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                remaining_str = str(timedelta(seconds=int(remaining))) if progress < 1.0 else "0:00:00"
                
                log_msg = f"Progresso: {progress*100:.1f}% ({state.global_step}/{state.max_steps}) " \
                          f"Decorrido: {elapsed_str}, Restante: {remaining_str}"
                
                # Adicionar informações de memória se monitoramento ativado
                if self.monitor_memory:
                    self._log_resource_usage()
                    
                logger.info(log_msg)
            
            self.last_log_time = current_time
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Chamado quando uma época termina"""
        logger.info(f"Concluída época {state.epoch}/{args.num_train_epochs}")
        
        # Registrar uso de recursos após cada época
        if self.monitor_memory:
            self._log_resource_usage()
        
    def on_train_end(self, args, state, control, **kwargs):
        """Chamado quando o treinamento termina"""
        total_time = time.time() - self.start_time
        logger.info(f"Treinamento concluído em {str(timedelta(seconds=int(total_time)))}")
        
        # Log final de recursos
        if self.monitor_memory:
            self._log_resource_usage(final=True)
    
    def _log_resource_usage(self, final=False):
        """Registra o uso atual de recursos"""
        try:
            # Uso de memória RAM
            ram = psutil.virtual_memory()
            ram_usage = ram.percent
            ram_used_gb = ram.used / (1024 ** 3)
            
            # Uso de GPU
            gpu_info = ""
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    gpu_mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    gpu_info += f" GPU {i}: {gpu_mem_alloc:.2f}GB alloc, {gpu_mem_reserved:.2f}GB reservado |"
                    
                    # Atualizar pico de uso
                    self.peak_memory_usage = max(self.peak_memory_usage, gpu_mem_alloc)
            
            # Uso de CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            resource_msg = f"Recursos: RAM: {ram_usage}% ({ram_used_gb:.2f}GB) | CPU: {cpu_percent}% |{gpu_info}"
            
            if final:
                logger.info(f"Uso final de recursos: {resource_msg}")
                if torch.cuda.is_available():
                    logger.info(f"Pico de uso de memória GPU: {self.peak_memory_usage:.2f}GB")
            else:
                logger.info(f"Uso atual de recursos: {resource_msg}")
                
        except Exception as e:
            logger.error(f"Erro ao monitorar recursos: {e}")