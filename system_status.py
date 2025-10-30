#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Status do Sistema Luna
Fornece informações completas sobre o estado atual do sistema
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SystemStatus:
    """Verifica e reporta o status completo do sistema Luna"""
    
    def __init__(self):
        self.status = {}
        
    def check_python_environment(self):
        """Verifica o ambiente Python"""
        logger.info("🐍 Ambiente Python")
        logger.info("-" * 50)
        logger.info(f"Versão: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        logger.info(f"Executável: {sys.executable}")
        logger.info(f"Plataforma: {sys.platform}")
        logger.info("")
        
    def check_dependencies(self):
        """Verifica dependências instaladas"""
        logger.info("📦 Dependências")
        logger.info("-" * 50)
        
        deps = {
            'torch': 'PyTorch',
            'transformers': 'HuggingFace Transformers',
            'nltk': 'NLTK',
            'stanza': 'Stanza',
            'sentence_transformers': 'Sentence Transformers',
            'flask': 'Flask',
            'flask_cors': 'Flask-CORS',
            'wandb': 'Weights & Biases',
            'numpy': 'NumPy',
            'pandas': 'Pandas',
            'faiss': 'FAISS',
            'optuna': 'Optuna (AutoML)'
        }
        
        installed = []
        missing = []
        
        for module, name in deps.items():
            try:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'desconhecida')
                logger.info(f"  ✓ {name:30s} {version}")
                installed.append(name)
            except ImportError:
                logger.info(f"  ✗ {name:30s} NÃO INSTALADO")
                missing.append(name)
        
        logger.info("")
        logger.info(f"Instaladas: {len(installed)}/{len(deps)}")
        
        if missing:
            logger.warning(f"Faltando: {', '.join(missing)}")
        
        logger.info("")
        
    def check_hardware(self):
        """Verifica hardware disponível"""
        logger.info("🖥️  Hardware")
        logger.info("-" * 50)
        
        try:
            import torch
            
            # CPU
            logger.info(f"CPU: {torch.get_num_threads()} threads")
            
            # GPU
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA disponível")
                logger.info(f"  Versão CUDA: {torch.version.cuda}")
                logger.info(f"  GPUs disponíveis: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            else:
                logger.info("✗ CUDA não disponível (CPU apenas)")
            
        except ImportError:
            logger.info("PyTorch não instalado - não é possível verificar hardware")
        
        logger.info("")
    
    def check_models(self):
        """Verifica modelos disponíveis"""
        logger.info("🤖 Modelos")
        logger.info("-" * 50)
        
        models_dir = Path("models")
        
        if not models_dir.exists():
            logger.info("Diretório de modelos não existe")
            logger.info("")
            return
        
        models = [d for d in models_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not models:
            logger.info("Nenhum modelo encontrado")
        else:
            logger.info(f"Modelos encontrados: {len(models)}")
            logger.info("")
            
            for model_dir in models:
                logger.info(f"  📁 {model_dir.name}")
                
                # Verificar arquivos principais
                config_file = model_dir / "config.json"
                tokenizer_dir = model_dir / "tokenizer"
                
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        logger.info(f"    ✓ Configuração presente")
                        if 'hidden_size' in config:
                            logger.info(f"      Hidden size: {config['hidden_size']}")
                        if 'num_hidden_layers' in config:
                            logger.info(f"      Camadas: {config['num_hidden_layers']}")
                    except:
                        logger.info(f"    ⚠ Configuração corrompida")
                else:
                    logger.info(f"    ✗ Configuração ausente")
                
                if tokenizer_dir.exists():
                    logger.info(f"    ✓ Tokenizador presente")
                else:
                    logger.info(f"    ✗ Tokenizador ausente")
                
                logger.info("")
        
    def check_data(self):
        """Verifica dados disponíveis"""
        logger.info("📊 Dados")
        logger.info("-" * 50)
        
        data_dirs = {
            'data/train': 'Treinamento',
            'data/valid': 'Validação',
            'data/rag_index': 'RAG Index'
        }
        
        for dir_path, label in data_dirs.items():
            path = Path(dir_path)
            if path.exists():
                files = list(path.rglob('*'))
                data_files = [f for f in files if f.is_file() and not f.name.startswith('.')]
                logger.info(f"  {label:15s}: {len(data_files)} arquivos")
            else:
                logger.info(f"  {label:15s}: Diretório não existe")
        
        logger.info("")
    
    def check_configuration(self):
        """Verifica arquivos de configuração"""
        logger.info("⚙️  Configuração")
        logger.info("-" * 50)
        
        config_files = [
            ('config/luna_default_config.json', 'Configuração padrão'),
            ('config/CONFIG_GUIDE.md', 'Guia de configuração'),
            ('.env', 'Variáveis de ambiente')
        ]
        
        for file_path, label in config_files:
            path = Path(file_path)
            if path.exists():
                logger.info(f"  ✓ {label}")
            else:
                logger.info(f"  ✗ {label:30s} (não encontrado)")
        
        logger.info("")
        
        # Verificar variáveis de ambiente Luna
        env_vars = {k: v for k, v in os.environ.items() if k.startswith('LUNA_')}
        if env_vars:
            logger.info("  Variáveis de ambiente configuradas:")
            for key, value in env_vars.items():
                # Ocultar valores sensíveis
                display_value = value if len(value) < 50 else value[:47] + "..."
                logger.info(f"    {key} = {display_value}")
        else:
            logger.info("  Nenhuma variável de ambiente LUNA_* configurada")
        
        logger.info("")
    
    def check_logs(self):
        """Verifica logs do sistema"""
        logger.info("📝 Logs")
        logger.info("-" * 50)
        
        logs_dir = Path("logs")
        
        if not logs_dir.exists():
            logger.info("Diretório de logs não existe")
            logger.info("")
            return
        
        log_files = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not log_files:
            logger.info("Nenhum arquivo de log encontrado")
        else:
            logger.info(f"Arquivos de log: {len(log_files)}")
            
            # Mostrar os 5 mais recentes
            for log_file in log_files[:5]:
                size_kb = log_file.stat().st_size / 1024
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                logger.info(f"  {log_file.name:30s} {size_kb:6.1f} KB  {mtime.strftime('%Y-%m-%d %H:%M')}")
        
        logger.info("")
    
    def check_features(self):
        """Verifica funcionalidades disponíveis"""
        logger.info("✨ Funcionalidades")
        logger.info("-" * 50)
        
        features = {
            'MoE': 'src/models/moe.py',
            'State-Space Layers': 'src/models/growing_network.py',
            'HyperNetworks': 'src/models/hypernet.py',
            'RAG': 'src/models/rag_retriever.py',
            'Chat Terminal': 'src/chat/luna_chat.py',
            'Chat Web': 'src/web/app.py',
            'Proactive Messaging': 'src/chat/proactive_messenger.py',
            'Feedback System': 'src/models/feedback_system.py',
            'Memory System': 'src/models/memory_system.py',
            'AutoML': 'src/optimization/automl_hyperparams.py'
        }
        
        available = 0
        for name, path in features.items():
            if Path(path).exists():
                logger.info(f"  ✓ {name}")
                available += 1
            else:
                logger.info(f"  ✗ {name}")
        
        logger.info("")
        logger.info(f"Disponíveis: {available}/{len(features)}")
        logger.info("")
    
    def calculate_system_score(self):
        """Calcula uma pontuação geral do sistema"""
        logger.info("📊 Pontuação do Sistema")
        logger.info("-" * 50)
        
        scores = {
            'Dependências': 0,
            'Hardware': 0,
            'Modelos': 0,
            'Dados': 0,
            'Configuração': 0,
            'Funcionalidades': 0
        }
        
        # Dependências (peso: 20%)
        try:
            deps = ['torch', 'transformers', 'nltk', 'flask']
            installed = sum(1 for d in deps if __import__(d, fromlist=['']))
            scores['Dependências'] = (installed / len(deps)) * 20
        except:
            scores['Dependências'] = 0
        
        # Hardware (peso: 15%)
        try:
            import torch
            scores['Hardware'] = 15 if torch.cuda.is_available() else 10
        except:
            scores['Hardware'] = 5
        
        # Modelos (peso: 20%)
        models_dir = Path("models")
        if models_dir.exists():
            models = [d for d in models_dir.iterdir() if d.is_dir()]
            scores['Modelos'] = min((len(models) / 3) * 20, 20)
        
        # Dados (peso: 15%)
        train_dir = Path("data/train")
        if train_dir.exists():
            files = list(train_dir.rglob('*'))
            scores['Dados'] = min((len(files) / 10) * 15, 15)
        
        # Configuração (peso: 15%)
        config_file = Path("config/luna_default_config.json")
        scores['Configuração'] = 15 if config_file.exists() else 5
        
        # Funcionalidades (peso: 15%)
        feature_files = [
            'src/models/moe.py',
            'src/models/rag_retriever.py',
            'src/chat/luna_chat.py',
            'src/web/app.py'
        ]
        available = sum(1 for f in feature_files if Path(f).exists())
        scores['Funcionalidades'] = (available / len(feature_files)) * 15
        
        # Total
        total = sum(scores.values())
        
        for category, score in scores.items():
            bar_length = int(score / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            logger.info(f"{category:18s} [{bar}] {score:5.1f}%")
        
        logger.info("")
        logger.info(f"{'PONTUAÇÃO TOTAL':18s} {total:5.1f}/100")
        
        # Classificação
        if total >= 90:
            classification = "🏆 EXCELENTE - Sistema pronto para produção!"
        elif total >= 75:
            classification = "✓ BOM - Sistema funcional com pequenas melhorias"
        elif total >= 50:
            classification = "⚠ ACEITÁVEL - Sistema precisa de melhorias"
        else:
            classification = "✗ INSUFICIENTE - Sistema precisa de trabalho significativo"
        
        logger.info(classification)
        logger.info("")
        
        return total
    
    def run_full_check(self):
        """Executa verificação completa do sistema"""
        logger.info("=" * 60)
        logger.info("LUNA GPT - STATUS DO SISTEMA")
        logger.info(f"Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        logger.info("")
        
        self.check_python_environment()
        self.check_dependencies()
        self.check_hardware()
        self.check_models()
        self.check_data()
        self.check_configuration()
        self.check_logs()
        self.check_features()
        
        score = self.calculate_system_score()
        
        logger.info("=" * 60)
        logger.info("Verificação concluída!")
        logger.info("=" * 60)
        
        return score


if __name__ == "__main__":
    status = SystemStatus()
    score = status.run_full_check()
    
    # Exit code based on score
    if score >= 75:
        sys.exit(0)
    else:
        sys.exit(1)
