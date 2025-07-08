import sys
import subprocess
import importlib
import logging

logger = logging.getLogger(__name__)

def check_and_install_dependencies():
    """Verifica e instala automaticamente as dependências faltantes."""
    required_packages = {
        'torch': 'torch>=2.0.0',
        'transformers': 'transformers>=4.30.0,<5.0.0',
        'tokenizers': 'tokenizers>=0.13.0',
        'tqdm': 'tqdm>=4.65.0',
        'regex': 'regex>=2023.6.3',
        'numpy': 'numpy>=1.24.0',
        'sentencepiece': 'sentencepiece>=0.1.99',
        'peft': 'peft>=0.4.0',
        'evaluate': 'evaluate>=0.4.0',
        'datasets': 'datasets>=2.12.0',
        'accelerate': 'accelerate>=0.20.0',
        'scikit-learn': 'scikit-learn>=1.2.2',
        'sentence-transformers': 'sentence-transformers>=2.2.2',
        'psutil': 'psutil>=5.9.0',
        'wandb': 'wandb>=0.15.0'
    }
    
    missing_packages = []
    
    for package, install_spec in required_packages.items():
        try:
            importlib.import_module(package)
            logger.debug(f"Dependência encontrada: {package}")
        except ImportError:
            logger.info(f"Dependência faltando: {package}")
            missing_packages.append(install_spec)
    
    if missing_packages:
        logger.info(f"Instalando {len(missing_packages)} dependências faltantes...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Instalado com sucesso: {package}")
            except subprocess.CalledProcessError:
                logger.error(f"Falha ao instalar {package}")
    
    return len(missing_packages)