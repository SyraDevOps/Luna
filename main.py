#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LunaGPT - Sistema de Diálogo Adaptativo e Dinâmico

Descrição:
    Sistema de IA conversacional com recursos avançados:
      - Camada State-Space para adaptação dinâmica
      - Salvamento multiformato (incluindo exportação ONNX)
      - Proatividade contextual com sugestões baseadas em regex
      - Otimizações no fluxo de treinamento e callbacks
"""

import os
import argparse
import logging
import glob
from pathlib import Path
import importlib
import nltk
nltk.download('punkt')

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
        gitkeep_file = os.path.join(directory, ".gitkeep")
        if not os.path.exists(gitkeep_file):
            with open(gitkeep_file, "w") as f:
                f.write("")

def load_data_from_directory(directory):
    data = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as file:
                data.extend(file.readlines())
    if not data:
        logger.warning(f"Nenhum dado encontrado em: {directory}")
        return []
    logger.info(f"Carregados {len(data)} linhas de {directory}")
    return [line.strip() for line in data if line.strip()]

def create_model(args, config):
    """Criar novo modelo e ambiente com dataset inicial"""
    model_name = args.model  # Alterado de args.name para args.model
    logger.info(f"Criando novo modelo: {model_name}")
    
    # Carregar dados iniciais para o tokenizer
    train_data = []
    if args.train_data:
        train_data, _ = load_data_from_patterns(args.train_data, auto_split=False)
    
    if not train_data:
        logger.warning("Nenhum dado de treinamento fornecido. Usando amostra padrão para inicializar o tokenizer.")
        train_data = [
            "Este é um exemplo de texto em português para treinar o modelo.",
            "O sistema LunaGPT é projetado para diálogo adaptativo em português.",
            "Processamento de linguagem natural é uma área fascinante da inteligência artificial."
        ]
    
    logger.info(f"Utilizando {len(train_data)} amostras para inicialização do tokenizer")
    
    # Treinar tokenizer
    tokenizer = LunaTokenizer(config)
    tokenizer.train_and_save(train_data, os.path.join("models", model_name, "tokenizer"))
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
        train_patterns = args.train_data
        logger.info(f"Usando padrões de dados personalizados: {train_patterns}")
    else:
        # Padrões padrão para vários formatos de arquivo
        train_patterns = [
            "data/train/*.txt", 
            "data/train/*.csv", 
            "data/train/*.json",
            "data/train/*.jsonl",
            "data/train/*.pdf",
            "data/train/*.docx"
        ]
        logger.info(f"Usando padrões de dados padrão: {train_patterns}")
    
    if args.valid_data:
        valid_patterns = args.valid_data
        auto_split = False
        logger.info(f"Usando padrões de validação personalizados: {valid_patterns}")
    else:
        # Padrões padrão para vários formatos de arquivo
        valid_patterns = [
            "data/valid/*.txt", 
            "data/valid/*.csv", 
            "data/valid/*.json",
            "data/valid/*.jsonl",
            "data/valid/*.pdf",
            "data/valid/*.docx"
        ]
        # Se não houver arquivo de validação, dividimos os dados de treino
        files_exist = any(len(glob.glob(pattern)) > 0 for pattern in valid_patterns)
        auto_split = not files_exist
        if auto_split:
            logger.info("Sem dados de validação explícitos. Dividindo dados de treino automaticamente.")
        else:
            logger.info(f"Usando padrões de validação padrão: {valid_patterns}")
    
    # Carregar dados
    train_data, auto_valid_data = load_data_from_patterns(train_patterns, auto_split=auto_split)
    
    if not auto_split:
        valid_data = load_data_from_patterns(valid_patterns, auto_split=False)[0]
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
        use_wandb = initialize_wandb(
            config=vars(config),
            run_name=f"train_{model_name}",
            project_name="lunagpt"
        )
    
    # Carregar dados
    train_data = []
    valid_data = []
    
    if args.train_data:
        train_data, valid_data = load_data_from_patterns(args.train_data)
    
    if not train_data:
        logger.error("Nenhum dado de treinamento fornecido. Use --train_data.")
        return False
    
    # Inicializar trainer
    trainer = LunaTrainer(model_name, config)
    
    # Usar AutoML se solicitado
    if args.automl:
        logger.info("Iniciando otimização de hiperparâmetros com AutoML")
        automl_config = {
            "study_name": f"lunagpt_automl_{model_name}",
            "n_trials": args.automl_trials,
            "metric": "validation_loss",
            "direction": "minimize"
        }
        result = trainer.train_with_automl(
            train_data, 
            valid_data, 
            automl_config=automl_config,
            use_wandb=use_wandb, 
            num_train_epochs=args.epochs
        )
    else:
        # Treinar modelo com ou sem otimização dinâmica
        result = trainer.train_supervised(
            train_data, 
            valid_data, 
            use_wandb=use_wandb, 
            num_train_epochs=args.epochs,
            dynamic_hp_opt=args.dynamic_hp_opt
        )
    
    if result and result.get("success", False):
        logger.info(f"Treinamento do modelo {model_name} concluído com sucesso.")
        return True
    else:
        logger.error(f"Treinamento do modelo {model_name} falhou.")
        return False

def chat_with_model(args, config):
    """Iniciar sessão de chat com modelo treinado"""
    model_name = args.model
    logger.info(f"Iniciando chat com o modelo: {model_name}")
    
    # Verificar se o modelo existe
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        logger.error(f"Modelo {model_name} não encontrado em {model_dir}")
        return
    
    # Ajustar configuração para modo leve se especificado
    if hasattr(args, 'lightweight') and args.lightweight:
        logger.info("Modo leve ativado: ajustando configurações para menor uso de memória")
        # Reduzir configurações
        config.training.per_device_train_batch_size = 1
        config.training.per_device_eval_batch_size = 1
        config.training.gradient_accumulation_steps = 8
        if hasattr(config.model, 'num_hidden_layers'):
            config.model.num_hidden_layers = min(config.model.num_hidden_layers, 4)
        if hasattr(config.model, 'hidden_size'):
            config.model.hidden_size = min(config.model.hidden_size, 256)
    
    # Forçar dispositivo específico se especificado
    if hasattr(args, 'device') and args.device != "auto":
        os.environ['FORCE_DEVICE'] = args.device
        if args.device == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = ""  # Desabilitar CUDA
        logger.info(f"Dispositivo forçado: {args.device}")
    
    # Configurar persona se especificada
    persona = args.persona if args.persona else config.persona.default
    
    # Inicializar chat
    chat_instance = LunaChat(model_name, config, persona=persona)
    
    # Iniciar chat com contexto inicial se especificado
    initial_context = args.context if args.context else ""
    chat_instance.chat(initial_context=initial_context)
    
    logger.info("Sessão de chat encerrada")

def refine_model(args, config):
    """Refinar modelo com feedback"""
    model_name = args.model
    logger.info(f"Iniciando refinamento do modelo: {model_name}")
    
    # Ajustar configuração para modo leve se especificado
    if hasattr(args, 'lightweight') and args.lightweight:
        logger.info("Modo leve ativado: ajustando configurações para menor uso de memória")
        # Reduzir configurações
        config.training.per_device_train_batch_size = 1
        config.training.per_device_eval_batch_size = 1
        config.training.gradient_accumulation_steps = 8
        if hasattr(config.model, 'num_hidden_layers'):
            config.model.num_hidden_layers = min(config.model.num_hidden_layers, 4)
        if hasattr(config.model, 'hidden_size'):
            config.model.hidden_size = min(config.model.hidden_size, 256)
    
    # Forçar dispositivo específico se especificado
    if hasattr(args, 'device') and args.device != "auto":
        os.environ['FORCE_DEVICE'] = args.device
        if args.device == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = ""  # Desabilitar CUDA
        logger.info(f"Dispositivo forçado: {args.device}")
    
    # Carregar sistema de feedback
    feedback_system = FeedbackSystem(config)
    
    if not feedback_system.needs_update():
        logger.warning("Feedback insuficiente para refinamento. Necessário pelo menos "
                      f"{config.feedback.min_samples_for_update} amostras.")
        return
    
    # Inicializar trainer
    trainer = LunaTrainer(model_name, config)
    
    # Atualizar modelo com feedback
    trainer.update_with_feedback()
    
    logger.info(f"Refinamento do modelo {model_name} concluído")

def manage_memory(args):
    """Gerencia o sistema de memória de um modelo"""
    model_name = args.model
    action = args.action
    
    # Verificar se o modelo existe
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        logger.error(f"Modelo {model_name} não encontrado")
        return
    
    # Carregar configuração
    config = Config()
    
    # Inicializar sistema de memória
    from src.models.memory_system import MemorySystem
    memory = MemorySystem(model_name, config)
    
    if action == 'stats':
        # Mostrar estatísticas
        stats = memory.get_memory_statistics()
        print("\n===== Estatísticas do Sistema de Memória =====")
        print(f"Modelo: {model_name}")
        print(f"Memórias episódicas: {stats['episodic_count']}")
        print(f"Memórias semânticas: {stats['semantic_count']}")
        print(f"Total de memórias: {stats['total_memories']}")
        print(f"Tipo de índice: {stats['index_type']}")
        print(f"Dimensão de embeddings: {stats['embedding_dimension']}")
        print("============================================\n")
        
    elif action == 'clear':
        # Pedir confirmação
        confirm = input(f"Tem certeza que deseja limpar TODAS as memórias do modelo {model_name}? (s/N): ")
        if confirm.lower() == 's':
            # Limpar memórias
            memory_dir = os.path.join("models", model_name, "memory")
            if os.path.exists(memory_dir):
                import shutil
                shutil.rmtree(memory_dir)
                os.makedirs(memory_dir, exist_ok=True)
                print(f"Memórias do modelo {model_name} foram limpas com sucesso.")
            else:
                print(f"O modelo {model_name} não possui memórias.")
        else:
            print("Operação cancelada.")
            
    elif action == 'import':
        # Importar documentos
        if not args.import_dir:
            logger.error("Diretório de importação não especificado (--import_dir)")
            return
            
        import_dir = args.import_dir
        if not os.path.exists(import_dir):
            logger.error(f"Diretório {import_dir} não encontrado")
            return
            
        # Verificar se LlamaIndex está disponível
        try:
            from src.models.knowledge_index import KnowledgeIndex
            knowledge_index = KnowledgeIndex(model_name)
            count = knowledge_index.import_from_files(import_dir)
            print(f"Importados {count} documentos para o índice de conhecimento.")
        except ImportError:
            # Fallback para importação básica
            from src.utils.file_utils import load_data_from_patterns
            patterns = [os.path.join(import_dir, "*.txt"), 
                       os.path.join(import_dir, "*.pdf"),
                       os.path.join(import_dir, "*.docx")]
            texts = load_data_from_patterns(patterns)
            for text in texts:
                memory.add_semantic_memory(text, metadata={"source": "import", "path": import_dir})
            print(f"Importados {len(texts)} textos para a memória semântica.")
            
    # Se houver uma consulta para teste, mostrar recuperação
    if args.query:
        print(f"\nTestando recuperação para consulta: '{args.query}'")
        relevant = memory.retrieve_relevant_memories(args.query, top_k=3)
        print(f"Encontradas {len(relevant)} memórias relevantes:")
        for i, mem in enumerate(relevant, 1):
            print(f"\n{i}. {mem.content[:100]}...")

def manage_tokens(args):
    """Gerencia tokens adaptativos"""
    from src.tools.token_analyzer import analyze_tokens
    
    if args.analyze:
        analyze_tokens(args.model, args.min_freq, args.max_tokens, args.add)
    else:
        print("Especifique uma ação para gerenciar tokens (--analyze)")

def run_tests():
    """Executar testes unitários"""
    logger.info("Executando testes unitários...")
    import unittest
    from src.tests.test_all import run_all_tests
    result = run_all_tests()
    logger.info(f"Resultado dos testes: {'Sucesso' if result == 0 else 'Falha'}")
    return result

def parse_args():
    parser = argparse.ArgumentParser(description='LunaGPT: Sistema de Diálogo Neural Adaptativo')
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis', required=True)
    
    # Parser para criar modelo
    create_parser = subparsers.add_parser('create', help='Criar novo modelo')
    create_parser.add_argument('--model', type=str, required=True, help='Nome do modelo a ser criado')
    create_parser.add_argument('--base', type=str, default=None, help='Modelo base para inicialização')
    create_parser.add_argument('--config', type=str, default=None, help='Caminho para arquivo de configuração')
    create_parser.add_argument('--train_data', type=str, default=None, help='Padrão de caminhos para dados de treinamento')
    
    # Parser para treinar modelo
    train_parser = subparsers.add_parser('train', help='Treinar modelo existente')
    train_parser.add_argument('--model', type=str, required=True, help='Nome do modelo a ser treinado')
    train_parser.add_argument('--train_data', type=str, default=None, help='Padrão de caminhos para dados de treinamento')
    train_parser.add_argument('--epochs', type=int, default=None, help='Número de épocas de treinamento')
    train_parser.add_argument('--dynamic_hp_opt', action='store_true', help='Ativar otimização dinâmica de hiperparâmetros')
    train_parser.add_argument('--automl', action='store_true', help='Usar AutoML para otimização de hiperparâmetros')
    train_parser.add_argument('--automl_trials', type=int, default=10, help='Número de tentativas para AutoML')
    
    # Comando 'chat'
    parser_chat = subparsers.add_parser("chat", help="Inicia modo de chat")
    parser_chat.add_argument("--model", type=str, required=True, help="Nome do modelo para chat")
    parser_chat.add_argument("--context", type=str, help="Contexto inicial")
    parser_chat.add_argument("--persona", type=str, choices=["tecnico", "casual", "formal"], 
                           help="Estilo de resposta")
    
    # Comando 'refine'
    parser_refine = subparsers.add_parser("refine", help="Realiza refinamento contínuo com feedback")
    parser_refine.add_argument("--model", type=str, required=True, help="Nome do modelo a refinar")
    
    # Comando 'memory'
    memory_parser = subparsers.add_parser('memory', help='Gerenciar sistema de memória')
    memory_parser.add_argument('--model', required=True, help='Nome do modelo')
    memory_parser.add_argument('--action', choices=['stats', 'clear', 'import'], required=True, 
                         help='Ação a executar no sistema de memória')
    memory_parser.add_argument('--import_dir', help='Diretório para importar documentos (para action=import)')
    memory_parser.add_argument('--query', help='Consulta para testar recuperação de memória')
    
    # Comando 'tokens'
    tokens_parser = subparsers.add_parser('tokens', help='Gerenciar tokens adaptativos')
    tokens_parser.add_argument('--model', required=True, help='Nome do modelo')
    tokens_parser.add_argument('--analyze', action='store_true', help='Analisar tokens candidatos')
    tokens_parser.add_argument('--min_freq', type=int, default=10, help='Frequência mínima para tokens')
    tokens_parser.add_argument('--max_tokens', type=int, default=100, help='Máximo de tokens a considerar')
    tokens_parser.add_argument('--add', action='store_true', help='Adicionar tokens automaticamente')
    
    # Adicionar opções de performance para todos os comandos que usam modelo
    for cmd_parser in [train_parser, parser_chat, parser_refine, memory_parser, tokens_parser]:
        cmd_parser.add_argument("--lightweight", action="store_true", 
                            help="Modo leve para máquinas com recursos limitados")
        cmd_parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"],
                            default="auto", help="Dispositivo para execução (auto=detectar)")
        cmd_parser.add_argument("--disable_lightweight", action="store_true", 
                          help="Desativa a otimização automática para hardware leve")
    
    # Comando 'test'
    parser_test = subparsers.add_parser("test", help="Executa testes unitários")
    parser_test.add_argument("--minimal", action="store_true", 
                          help="Executar testes com configurações mínimas para economizar memória")
    
    return parser.parse_args()

def main():
    # Verificar dependências antes de iniciar
    from src.utils.dependency_utils import check_and_install_dependencies
    installed = check_and_install_dependencies()
    if installed > 0:
        logger.info(f"Instaladas {installed} dependências faltantes. Reinicializando componentes.")
        # Se instalou algo, deve reimportar os módulos que foram instalados
        importlib.invalidate_caches()
    
    # Configurar logging
    setup_logging()
    
    # Verificar diretórios
    ensure_data_directories()
    
    # Carregar configuração
    config = Config()
    
    # Parsear argumentos
    args = parse_args()
    
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
        run_tests()
    else:
        logger.error("Comando inválido. Use --help para ver os comandos disponíveis.")

if __name__ == "__main__":
    main()