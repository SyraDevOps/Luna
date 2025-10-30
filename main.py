#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LunaGPT - Sistema de Diálogo Adaptativo e Dinâmico
"""
import os
import sys
import argparse
import logging
import time
import nltk
from pathlib import Path

# Baixar recursos necessários do NLTK
try:
    nltk.download('punkt', quiet=True)
except:
    pass

from src.config.config import Config
from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer
from src.training.trainer import LunaTrainer
from src.chat.luna_chat import LunaChat
from src.utils.logging_utils import setup_logging
from src.models.feedback_system import FeedbackSystem
from src.utils.file_utils import load_data_from_patterns
from src.utils.hardware_utils import detect_hardware, setup_memory_efficient_training
from src.models.rag_retriever import RAGRetriever
from src.utils.wandb_utils import initialize_wandb

logger = logging.getLogger(__name__)

def ensure_data_directories():
    """Garante que as pastas necessárias existam no diretório do projeto"""
    directories = ["data/train", "data/valid", "models", "temp", "logs", "wandb"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Diretório verificado/criado: {directory}")
    
    # Criar arquivo .gitkeep para garantir que o diretório seja mantido no repositório
    for directory in directories:
        gitkeep_path = os.path.join(directory, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, "w") as f:
                f.write("")

def load_data_from_directory(directory):
    """Carrega dados de um diretório"""
    data = []
    if not os.path.exists(directory):
        logger.warning(f"Diretório não encontrado: {directory}")
        return data
        
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                from src.utils.file_utils import load_file
                file_data = load_file(filepath)
                data.extend(file_data)
            except Exception as e:
                logger.error(f"Erro ao carregar {filepath}: {str(e)}")
    
    if not data:
        logger.warning(f"Nenhum dado encontrado em {directory}")
    
    logger.info(f"Carregados {len(data)} arquivos de {directory}")
    return data

def apply_args_to_config(args, config):
    """Aplica argumentos da linha de comando à configuração"""
    # Aplicar configurações do modelo se create
    if hasattr(args, 'vocab_size') and args.vocab_size:
        config.tokenizer.vocab_size = args.vocab_size
        config.model.vocab_size = args.vocab_size
    if hasattr(args, 'hidden_size') and args.hidden_size:
        config.model.hidden_size = args.hidden_size
    if hasattr(args, 'num_layers') and args.num_layers:
        config.model.num_hidden_layers = args.num_layers
    if hasattr(args, 'use_moe') and args.use_moe:
        config.model.use_moe = True
    if hasattr(args, 'num_experts') and args.num_experts:
        config.model.num_experts = args.num_experts
    if hasattr(args, 'use_state_space') and args.use_state_space:
        config.model.use_state_space = True
    if hasattr(args, 'use_hypernet') and args.use_hypernet:
        config.model.use_hypernet = True
    
    # Aplicar configurações de treinamento
    if hasattr(args, 'epochs') and args.epochs:
        config.training.num_train_epochs = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if hasattr(args, 'weight_decay') and args.weight_decay:
        config.training.weight_decay = args.weight_decay
    if hasattr(args, 'warmup_ratio') and args.warmup_ratio:
        config.training.warmup_ratio = args.warmup_ratio
    if hasattr(args, 'gradient_accumulation') and args.gradient_accumulation:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if hasattr(args, 'max_grad_norm') and args.max_grad_norm:
        config.training.max_grad_norm = args.max_grad_norm
    if hasattr(args, 'use_wandb') and args.use_wandb:
        config.training.use_wandb = True
    if hasattr(args, 'use_curriculum') and args.use_curriculum:
        config.training.use_curriculum = True
    if hasattr(args, 'fp16') and args.fp16:
        config.training.fp16 = True
    if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
        config.training.gradient_checkpointing = True
    if hasattr(args, 'use_automl') and args.use_automl:
        config.optimization.use_automl = True
    if hasattr(args, 'automl_trials') and args.automl_trials:
        config.optimization.automl_trials = args.automl_trials
    
    # Aplicar configurações de RAG
    if hasattr(args, 'use_rag') and args.use_rag:
        config.rag.use_rag = True
    
    # Aplicar configurações de feedback
    if hasattr(args, 'min_samples') and args.min_samples:
        config.feedback.min_samples_for_update = args.min_samples
    if hasattr(args, 'quality_threshold') and args.quality_threshold:
        config.feedback.quality_threshold = args.quality_threshold
    
    # Configurações de timeout
    if hasattr(args, 'no_timeout') and args.no_timeout:
        # Desabilitar timeouts configurando valores muito altos
        logger.info("Timeouts desabilitados conforme solicitado")
        os.environ['LUNA_NO_TIMEOUT'] = 'true'
    
    return config

def create_model(args, config):
    """Criar novo modelo e ambiente com dataset inicial"""
    # Aplicar argumentos da linha de comando à configuração
    config = apply_args_to_config(args, config)
    
    model_name = args.model
    logger.info(f"Criando novo modelo: {model_name}")
    
    # Carregar dados iniciais para o tokenizer
    train_data = []
    if args.train_data:
        train_data, _ = load_data_from_patterns([args.train_data], auto_split=False)
    
    if not train_data:
        train_data = load_data_from_directory("data/train")
    
    if not train_data:
        # Dados de exemplo mínimos
        train_data = [
            "Olá! Como posso ajudá-lo hoje?",
            "Pergunta: O que é inteligência artificial? Resposta: IA é a simulação de processos de inteligência humana por máquinas.",
            "Estou aqui para conversar e responder suas perguntas."
        ]
        logger.warning("Usando dados de exemplo mínimos para o tokenizer")
    
    logger.info(f"Utilizando {len(train_data)} amostras para inicialização do tokenizer")
    
    # Treinar tokenizer
    tokenizer = LunaTokenizer(config)
    tokenizer_dir = os.path.join("models", model_name, "tokenizer")
    tokenizer.train_and_save(train_data, tokenizer_dir)
    tokenizer.configure_special_tokens()
    
    # Criar modelo do zero
    model = LunaModel.from_scratch(config.model)
    
    # Salvar modelo inicial
    model_dir = os.path.join("models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    
    logger.info(f"Modelo {model_name} criado com sucesso em {model_dir}")
    return model_name

def load_training_and_validation_data(args):
    """Carregar dados de treinamento e validação"""
    if args.train_data:
        train_patterns = [args.train_data]
    else:
        train_patterns = ["data/train/*.txt", "data/train/*.csv", "data/train/*.json", "data/train/*.pdf"]
    
    if args.valid_data:
        valid_patterns = [args.valid_data]
        auto_split = False
    else:
        valid_patterns = ["data/valid/*.txt", "data/valid/*.csv", "data/valid/*.json"]
        auto_split = True
    
    # Carregar dados
    train_data, auto_valid_data = load_data_from_patterns(train_patterns, auto_split=auto_split)
    
    if not auto_split:
        valid_data, _ = load_data_from_patterns(valid_patterns, auto_split=False)
    else:
        valid_data = auto_valid_data
    
    return train_data, valid_data

def train_model(args, config):
    """Treinar modelo existente com novos dados"""
    # Aplicar argumentos da linha de comando à configuração
    config = apply_args_to_config(args, config)
    
    model_name = args.model
    logger.info(f"Treinando modelo: {model_name}")
    
    # Inicializar wandb se habilitado na configuração
    use_wandb = config.training.use_wandb
    if use_wandb:
        initialize_wandb(config, f"train_{model_name}")
    
    # Carregar dados
    train_data, valid_data = load_training_and_validation_data(args)
    
    if not train_data:
        logger.error("Nenhum dado de treinamento encontrado!")
        return
    
    # Inicializar trainer
    trainer = LunaTrainer(model_name, config)
    
    # Treinar modelo
    try:
        result = trainer.train_supervised(
            train_data=train_data,
            valid_data=valid_data,
            use_wandb=use_wandb,
            num_train_epochs=args.epochs
        )
        
        if result.get("success"):
            logger.info("Treinamento concluído com sucesso!")
        else:
            logger.error(f"Erro durante treinamento: {result.get('error', 'Erro desconhecido')}")
            
    except Exception as e:
        logger.error(f"Erro durante treinamento: {str(e)}")

def chat_with_model(args, config):
    """Iniciar chat interativo com modelo"""
    model_name = args.model
    model_dir = os.path.join("models", model_name)
    
    if not os.path.exists(model_dir):
        logger.error(f"Modelo {model_name} não encontrado. Use 'create' primeiro.")
        return
    
    logger.info(f"Iniciando chat com modelo: {model_name}")
    
    try:
        # Inicializar chat
        chat = LunaChat(model_name, config, persona=args.persona)
        
        # Iniciar sessão interativa
        chat.chat()
        
    except KeyboardInterrupt:
        logger.info("Chat interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro durante chat: {str(e)}")

def refine_model(args, config):
    """Refinar modelo com base no feedback"""
    model_name = args.model
    logger.info(f"Refinando modelo: {model_name}")
    
    try:
        # Inicializar trainer
        trainer = LunaTrainer(model_name, config)
        
        # Atualizar com feedback
        result = trainer.update_with_feedback(use_wandb=config.training.use_wandb)
        
        if result.get("success"):
            logger.info("Refinamento concluído com sucesso!")
        else:
            logger.info("Nenhuma atualização necessária ou feedback insuficiente")
            
    except Exception as e:
        logger.error(f"Erro durante refinamento: {str(e)}")

def manage_memory(args):
    """Gerenciar sistema de memória"""
    from src.models.memory_system import MemorySystem
    
    model_name = args.model
    logger.info(f"Gerenciando memória do modelo: {model_name}")
    
    try:
        memory = MemorySystem(model_name)
        
        if args.action == "stats":
            stats = memory.get_memory_statistics()
            logger.info(f"Estatísticas de memória: {stats}")
            
        elif args.action == "save":
            memory.save()
            logger.info("Memória salva com sucesso")
            
        elif args.action == "clear":
            # Implementar limpeza se necessário
            logger.info("Funcionalidade de limpeza ainda não implementada")
            
    except Exception as e:
        logger.error(f"Erro ao gerenciar memória: {str(e)}")

def manage_tokens(args):
    """Gerenciar tokens adaptativos"""
    model_name = args.model
    logger.info(f"Gerenciando tokens do modelo: {model_name}")
    
    try:
        if args.action == "analyze":
            from src.tools.token_analyzer import analyze_tokens
            analyze_tokens(
                model_name, 
                min_frequency=args.min_freq, 
                max_tokens=args.max_tokens,
                auto_add=args.auto_add
            )
        else:
            logger.error(f"Ação desconhecida: {args.action}")
            
    except Exception as e:
        logger.error(f"Erro ao gerenciar tokens: {str(e)}")

def start_web_interface(args, config):
    """Iniciar interface web"""
    model_name = args.model
    logger.info(f"Iniciando interface web para modelo: {model_name}")
    
    try:
        from src.web.app import create_app
        app = create_app(model_name, config)
        app.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f"Erro ao iniciar interface web: {str(e)}")

def manage_config(args, config):
    """Gerenciar configuração do sistema"""
    try:
        if args.action == "show":
            summary = config.get_summary()
            logger.info("=== Resumo da Configuração Atual ===")
            import json
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            
        elif args.action == "save":
            output_path = args.output or "config/luna_config.json"
            config.save_to_file(output_path)
            logger.info(f"Configuração salva em: {output_path}")
            
        elif args.action == "validate":
            config._validate_config()
            logger.info("Configuração validada com sucesso!")
            summary = config.get_summary()
            logger.info(f"Resumo: {summary}")
            
    except Exception as e:
        logger.error(f"Erro ao gerenciar configuração: {str(e)}")

def run_tests():
    """Executar testes do sistema"""
    logger.info("Executando testes do sistema...")
    
    try:
        from src.tests.test_all import run_all_tests
        result = run_all_tests()
        
        if result == 0:
            logger.info("Todos os testes passaram!")
        else:
            logger.error("Alguns testes falharam")
            
        return result
        
    except Exception as e:
        logger.error(f"Erro ao executar testes: {str(e)}")
        return 1

def parse_args():
    """Analisar argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description="LunaGPT - Sistema de Diálogo Adaptativo e Configurável")
    
    # Argumentos globais
    parser.add_argument("--config", type=str, help="Caminho para arquivo de configuração JSON")
    parser.add_argument("--no-timeout", action="store_true", help="Desabilitar timeouts em operações")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Nível de logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    # Comando create
    create_parser = subparsers.add_parser("create", help="Criar novo modelo")
    create_parser.add_argument("--model", required=True, help="Nome do modelo")
    create_parser.add_argument("--train-data", help="Padrão de arquivos de treino")
    create_parser.add_argument("--vocab-size", type=int, help="Tamanho do vocabulário")
    create_parser.add_argument("--hidden-size", type=int, help="Tamanho da camada oculta")
    create_parser.add_argument("--num-layers", type=int, help="Número de camadas")
    create_parser.add_argument("--use-moe", action="store_true", help="Habilitar Mixture of Experts")
    create_parser.add_argument("--num-experts", type=int, help="Número de especialistas MoE")
    create_parser.add_argument("--use-state-space", action="store_true", help="Habilitar State-Space Layers")
    create_parser.add_argument("--use-hypernet", action="store_true", help="Habilitar HyperNetworks")
    
    # Comando train
    train_parser = subparsers.add_parser("train", help="Treinar modelo")
    train_parser.add_argument("--model", required=True, help="Nome do modelo")
    train_parser.add_argument("--train-data", help="Padrão de arquivos de treino")
    train_parser.add_argument("--valid-data", help="Padrão de arquivos de validação")
    train_parser.add_argument("--epochs", type=int, help="Número de épocas")
    train_parser.add_argument("--batch-size", type=int, help="Tamanho do batch")
    train_parser.add_argument("--learning-rate", type=float, help="Taxa de aprendizado")
    train_parser.add_argument("--weight-decay", type=float, help="Peso de decaimento")
    train_parser.add_argument("--warmup-ratio", type=float, help="Proporção de warmup")
    train_parser.add_argument("--gradient-accumulation", type=int, help="Passos de acumulação de gradiente")
    train_parser.add_argument("--max-grad-norm", type=float, help="Norma máxima do gradiente")
    train_parser.add_argument("--use-wandb", action="store_true", help="Usar Weights & Biases")
    train_parser.add_argument("--use-curriculum", action="store_true", help="Usar curriculum learning")
    train_parser.add_argument("--use-automl", action="store_true", help="Usar AutoML para otimização")
    train_parser.add_argument("--automl-trials", type=int, help="Número de trials do AutoML")
    train_parser.add_argument("--fp16", action="store_true", help="Usar precisão mista FP16")
    train_parser.add_argument("--gradient-checkpointing", action="store_true", help="Usar gradient checkpointing")
    
    # Comando chat
    chat_parser = subparsers.add_parser("chat", help="Chat interativo")
    chat_parser.add_argument("--model", required=True, help="Nome do modelo")
    chat_parser.add_argument("--persona", help="Persona para o chat")
    chat_parser.add_argument("--use-rag", action="store_true", help="Habilitar RAG")
    chat_parser.add_argument("--use-proactive", action="store_true", help="Habilitar mensagens proativas")
    chat_parser.add_argument("--max-history", type=int, help="Máximo de mensagens no histórico")
    chat_parser.add_argument("--temperature", type=float, help="Temperatura de geração")
    chat_parser.add_argument("--top-p", type=float, help="Top-p sampling")
    chat_parser.add_argument("--max-tokens", type=int, help="Máximo de tokens a gerar")
    
    # Comando web
    web_parser = subparsers.add_parser("web", help="Iniciar interface web")
    web_parser.add_argument("--model", required=True, help="Nome do modelo")
    web_parser.add_argument("--host", default="0.0.0.0", help="Host do servidor")
    web_parser.add_argument("--port", type=int, default=5000, help="Porta do servidor")
    web_parser.add_argument("--debug", action="store_true", help="Modo debug")
    
    # Comando refine
    refine_parser = subparsers.add_parser("refine", help="Refinar modelo com feedback")
    refine_parser.add_argument("--model", required=True, help="Nome do modelo")
    refine_parser.add_argument("--min-samples", type=int, help="Mínimo de amostras para refinar")
    refine_parser.add_argument("--quality-threshold", type=int, help="Threshold de qualidade")
    
    # Comando memory
    memory_parser = subparsers.add_parser("memory", help="Gerenciar memória")
    memory_parser.add_argument("--model", required=True, help="Nome do modelo")
    memory_parser.add_argument("--action", choices=["stats", "save", "clear"], 
                             default="stats", help="Ação a executar")
    
    # Comando tokens
    tokens_parser = subparsers.add_parser("tokens", help="Gerenciar tokens")
    tokens_parser.add_argument("--model", required=True, help="Nome do modelo")
    tokens_parser.add_argument("--action", choices=["analyze"], default="analyze")
    tokens_parser.add_argument("--min-freq", type=int, default=10, help="Frequência mínima")
    tokens_parser.add_argument("--max-tokens", type=int, default=100, help="Máximo de tokens")
    tokens_parser.add_argument("--auto-add", action="store_true", help="Adicionar automaticamente")
    
    # Comando test
    test_parser = subparsers.add_parser("test", help="Executar testes")
    test_parser.add_argument("--test-type", choices=["all", "unit", "integration", "performance"], 
                           default="all", help="Tipo de teste")
    
    # Comando config
    config_parser = subparsers.add_parser("config", help="Gerenciar configuração")
    config_parser.add_argument("--action", choices=["show", "save", "validate"], 
                             default="show", help="Ação a executar")
    config_parser.add_argument("--output", help="Arquivo de saída para salvar configuração")
    
    return parser.parse_args()

def main():
    """Função principal"""
    # Analisar argumentos primeiro para obter log-level
    args = parse_args()
    
    # Configurar logging com nível apropriado
    log_level = getattr(args, 'log_level', 'INFO')
    setup_logging(level=log_level)
    
    # Garantir estrutura de diretórios
    ensure_data_directories()
    
    # Configurar otimizações de memória
    setup_memory_efficient_training()
    
    if not args.command:
        logger.error("Nenhum comando especificado. Use --help para ver opções.")
        return 1
    
    # Carregar configuração
    config_path = getattr(args, 'config', None)
    config = Config(config_path=config_path)
    
    # Aplicar argumentos CLI à configuração
    config = apply_args_to_config(args, config)
    
    try:
        if args.command == "create":
            create_model(args, config)
            
        elif args.command == "train":
            train_model(args, config)
            
        elif args.command == "chat":
            chat_with_model(args, config)
            
        elif args.command == "web":
            start_web_interface(args, config)
            
        elif args.command == "refine":
            refine_model(args, config)
            
        elif args.command == "memory":
            manage_memory(args)
            
        elif args.command == "tokens":
            manage_tokens(args)
            
        elif args.command == "config":
            manage_config(args, config)
            
        elif args.command == "test":
            return run_tests()
            
        else:
            logger.error(f"Comando desconhecido: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operação cancelada pelo usuário")
        return 0
        
    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
