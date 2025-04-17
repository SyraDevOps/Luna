import os
import logging
import sys
from datetime import datetime
from typing import Optional
import traceback

# Configuração inicial do logger (será reconfigurado por setup_logging)
logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO, log_dir: Optional[str] = None):
    """Configura o sistema de logging"""
    # Criar diretório de logs se não existir
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), 'logs')
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"luna_{timestamp}.log")
    
    # Configurar handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler de arquivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Handler de console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Registrar início do log
    logging.getLogger(__name__).info(f"Log iniciado em {log_file}")
    
    cleanup_old_logs(log_dir=log_dir)
    
    return log_file

def cleanup_old_logs(log_dir: str, max_logs=50, max_days=30):
    """
    Remove logs antigos para economizar espaço
    
    Args:
        log_dir: Diretório onde os logs estão armazenados
        max_logs: Número máximo de arquivos de log a manter
        max_days: Idade máxima dos logs em dias
    """
    try:
        if not os.path.exists(log_dir):
            return
            
        log_files = []
        for f in os.listdir(log_dir):
            full_path = os.path.join(log_dir, f)
            if os.path.isfile(full_path) and f.startswith("luna_") and f.endswith(".log"):
                log_files.append((full_path, os.path.getmtime(full_path)))
                
        # Ordenar por data (mais recentes primeiro)
        log_files.sort(key=lambda x: x[1], reverse=True)
        
        # Remover excesso de logs
        if len(log_files) > max_logs:
            for path, _ in log_files[max_logs:]:
                try:
                    os.remove(path)
                    logger.debug(f"Log antigo removido: {path}")
                except Exception:
                    pass
                    
        # Remover logs muito antigos
        cutoff_time = datetime.now().timestamp() - (max_days * 86400)
        for path, mtime in log_files:
            if mtime < cutoff_time:
                try:
                    os.remove(path)
                    logger.debug(f"Log expirado removido: {path}")
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"Erro ao limpar logs antigos: {e}")

def log_exception(e, context=""):
    """Registra uma exceção com contexto opcional"""
    logger.error(f"Erro no contexto [{context}]: {str(e)}")
    logger.debug(traceback.format_exc())