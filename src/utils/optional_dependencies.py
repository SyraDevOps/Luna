import os
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_optional_dependencies() -> Dict[str, Any]:
    deps = {}
    try:
        import stanza
        stanza.download('pt')  # Faz o download do modelo 'pt' automaticamente
        deps['nlp'] = stanza.Pipeline(lang='pt', processors='tokenize,pos', use_gpu=torch.cuda.is_available())
    except ImportError:
        logger.warning("Stanza não está instalado. Instalando automaticamente...")
        os.system("pip install stanza")
        try:
            import stanza
            stanza.download('pt')
            deps['nlp'] = stanza.Pipeline(lang='pt', processors='tokenize,pos', use_gpu=torch.cuda.is_available())
        except Exception as e:
            logger.error(f"Erro ao instalar ou configurar o Stanza: {e}")
            deps['nlp'] = None
    return deps