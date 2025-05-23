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

def create_model(args, config):
    """Criar novo modelo e ambiente com dataset inicial"""
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
    parser = argparse.ArgumentParser(description="LunaGPT - Sistema de Diálogo Adaptativo")
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    # Comando create
    create_parser = subparsers.add_parser("create", help="Criar novo modelo")
    create_parser.add_argument("--model", required=True, help="Nome do modelo")
    create_parser.add_argument("--train-data", help="Padrão de arquivos de treino")
    
    # Comando train
    train_parser = subparsers.add_parser("train", help="Treinar modelo")
    train_parser.add_argument("--model", required=True, help="Nome do modelo")
    train_parser.add_argument("--train-data", help="Padrão de arquivos de treino")
    train_parser.add_argument("--valid-data", help="Padrão de arquivos de validação")
    train_parser.add_argument("--epochs", type=int, default=3, help="Número de épocas")
    
    # Comando chat
    chat_parser = subparsers.add_parser("chat", help="Chat interativo")
    chat_parser.add_argument("--model", required=True, help="Nome do modelo")
    chat_parser.add_argument("--persona", default="casual", help="Persona para o chat")
    
    # Comando refine
    refine_parser = subparsers.add_parser("refine", help="Refinar modelo com feedback")
    refine_parser.add_argument("--model", required=True, help="Nome do modelo")
    
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
    
    return parser.parse_args()

def main():
    """Função principal"""
    # Configurar logging
    setup_logging()
    
    # Garantir estrutura de diretórios
    ensure_data_directories()
    
    # Configurar otimizações de memória
    setup_memory_efficient_training()
    
    # Analisar argumentos
    args = parse_args()
    
    if not args.command:
        logger.error("Nenhum comando especificado. Use --help para ver opções.")
        return 1
    
    # Carregar configuração
    config = Config()
    
    try:
        if args.command == "create":
            create_model(args, config)
            
        elif args.command == "train":
            train_model(args, config)
            
        elif args.command == "chat":
            chat_with_model(args, config)
            
        elif args.command == "refine":
            refine_model(args, config)
            
        elif args.command == "memory":
            manage_memory(args)
            
        elif args.command == "tokens":
            manage_tokens(args)
            
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
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
