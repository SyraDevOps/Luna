import os
import logging
import importlib.util

logger = logging.getLogger(__name__)

# ATENÇÃO: Definir o token diretamente no código representa um risco de segurança
# especialmente se o repositório for compartilhado ou público
# Use preferencialmente variáveis de ambiente ou o comando 'wandb login'
WANDB_API_KEY = "36043b774e05408203aaba632d2f2c50c83280c8"  # Substitua None pelo seu token, ex: "1a2b3c4d5e6f7g8h9i0j"

def is_wandb_available():
    """Verifica se o wandb está instalado"""
    return importlib.util.find_spec("wandb") is not None

def initialize_wandb(config, run_name=None, project_name="lunagpt"):
    """
    Inicializa o wandb com tratamento adequado de erros
    
    Args:
        config: Configuração do modelo
        run_name: Nome da execução (opcional)
        project_name: Nome do projeto no wandb (padrão: "lunagpt")
        
    Returns:
        bool: True se inicializado com sucesso, False caso contrário
    """
    if not is_wandb_available():
        logger.warning("Weights & Biases (wandb) não está instalado. Instalando...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb>=0.15.0"])
            logger.info("wandb instalado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao instalar wandb: {str(e)}")
            return False
    
    try:
        import wandb
        
        # Verificar se já existe uma execução ativa
        if wandb.run is not None:
            logger.info(f"Reutilizando execução wandb ativa: {wandb.run.name}")
            return True
        
        # Usar o token definido no código, se disponível
        if WANDB_API_KEY:
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
            logger.info("Usando token W&B definido no código")
            
        # Verificar se há API key configurada
        if "WANDB_API_KEY" not in os.environ and not os.path.exists(os.path.expanduser("~/.netrc")):
            logger.warning("API key do wandb não configurada. Execute 'wandb login' ou defina WANDB_API_KEY.")
            return False
            
        # Nome da execução padrão baseado na hora atual se não especificado
        if run_name is None:
            import datetime
            run_name = f"luna_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Inicializar wandb
        wandb.init(project=project_name, name=run_name, config=config)
        logger.info(f"Weights & Biases inicializado para projeto '{project_name}', execução '{run_name}'")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao inicializar wandb: {str(e)}")
        return False