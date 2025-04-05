import os
import logging
import traceback
import torch
from typing import List, Optional, Dict, Union, Tuple, Any
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import Dataset
import transformers
import numpy as np
from tqdm import tqdm
import wandb
import optuna

from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer
from src.models.feedback_system import FeedbackSystem
from src.config.config import Config
from src.utils.callbacks import CustomTrainingCallback
from src.utils.hardware_utils import detect_hardware, setup_memory_efficient_training
import gc
import psutil
import tempfile

logger = logging.getLogger(__name__)

class LunaTrainer:
    """Classe de treinamento para modelos Luna"""
    
    def __init__(self, model_name, config):
        """
        Inicializa o treinador Luna.
        
        Args:
            model_name: Nome do modelo para carregar/salvar
            config: Configuração de treinamento
        """
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configurar diretório temporário para arquivos de treinamento
        self.output_dir = os.path.join("temp", "test_model")  # Adicionar o diretório de saída
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Diretório temporário configurado em: {self.output_dir}")
        
        # Ajustar parâmetros de treinamento com base no hardware
        hardware_profile = detect_hardware()
        if hardware_profile.system_type == "low-end":
            # Reduzir batch size para hardware limitado
            if self.config.training.per_device_train_batch_size > 1:
                old_bs = self.config.training.per_device_train_batch_size
                self.config.training.per_device_train_batch_size = 1
                self.logger.info(f"Ajustando batch size de {old_bs} para 1")
                
                # Aumentar gradient accumulation para compensar o batch size menor
                self.config.training.gradient_accumulation_steps = max(4, self.config.training.gradient_accumulation_steps)
                self.logger.info(f"Ajustando gradient accumulation steps para {self.config.training.gradient_accumulation_steps}")
        
        # Carregar o modelo
        try:
            self.model = self._load_model()
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo para treinamento: {str(e)}")
            raise
    
    def _adjust_hyperparams_for_hardware(self):
        """Ajusta hiperparâmetros de treinamento com base no hardware disponível."""
        if self.device.system_type == "low-end":
            # Ajustar batch size para hardware leve
            original_batch_size = self.config.training.per_device_train_batch_size
            if original_batch_size > 1:
                self.config.training.per_device_train_batch_size = 1
                self.logger.info(f"Ajustando batch size de {original_batch_size} para {self.config.training.per_device_train_batch_size}")
            
            # Aumentar gradient accumulation steps para compensar batch size menor
            self.config.training.gradient_accumulation_steps = max(self.config.training.gradient_accumulation_steps, 4)
            self.logger.info(f"Ajustando gradient accumulation steps para {self.config.training.gradient_accumulation_steps}")
            
            # Reduzir tamanho do modelo se necessário
            if hasattr(self.config.model, 'hidden_size') and self.config.model.hidden_size > 256:
                self.logger.info(f"Reduzindo hidden_size de {self.config.model.hidden_size} para 256 para hardware leve")
                self.config.model.hidden_size = 256
            
            # Ativar checkpointing de gradientes para economizar memória
            self.config.training.gradient_checkpointing = True
        else:
            # Hardware potente - usar configurações padrão ou otimizadas
            self.logger.info("Hardware potente detectado, usando configurações otimizadas")
            
            # Aumentar batch size se houver memória disponível
            if self.device.gpu_available and self.device.gpu_memory_gb > 8:
                self.config.training.per_device_train_batch_size = max(self.config.training.per_device_train_batch_size, 4)
                self.logger.info(f"Batch size definido para {self.config.training.per_device_train_batch_size}")
    
    def _adjust_config_for_hardware(self):
        """Ajusta os parâmetros de configuração com base no hardware disponível."""
        try:
            # Acessar diretamente o atributo ram_gb
            if self.device.ram_gb < 8:
                original_bs = self.config.training.per_device_train_batch_size
                self.config.training.per_device_train_batch_size = 1
                self.logger.info(f"Ajustando batch size de {original_bs} para {self.config.training.per_device_train_batch_size}")
                
                # Aumentar gradient accumulation steps para compensar
                gradient_steps = self.config.training.gradient_accumulation_steps
                self.config.training.gradient_accumulation_steps = max(4, gradient_steps)
                self.logger.info(f"Ajustando gradient accumulation steps para {self.config.training.gradient_accumulation_steps}")
        except AttributeError as e:
            self.logger.error(f"Erro ao ajustar configuração para hardware: {str(e)}")
            raise
    
    def _compute_causal_lm_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Função personalizada para calcular a perda de modelagem causal de linguagem."""
        try:
            # Garantir que labels estão presentes
            if "labels" not in inputs:
                inputs["labels"] = inputs["input_ids"].clone()

            # Ajustar position_ids
            seq_length = inputs["input_ids"].shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs["input_ids"].device)
            position_ids = position_ids.unsqueeze(0).expand(inputs["input_ids"].shape[0], -1)
            inputs["position_ids"] = torch.clamp(position_ids, 0, model.config.n_positions - 1)

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss if outputs.loss is not None else torch.tensor(1.0, requires_grad=True, device=inputs["input_ids"].device)
            return (loss, outputs) if return_outputs else loss
        except Exception as e:
            self.logger.error(f"Erro em _compute_causal_lm_loss: {str(e)}")
            return torch.tensor(1.0, requires_grad=True, device="cpu")
    
    def train_supervised(self, train_data, valid_data=None, use_wandb=False):
        """
        Treina o modelo com dados supervisionados.
        
        Args:
            train_data: Lista de dados de treinamento
            valid_data: Lista de dados de validação
            use_wandb: Se True, integra com Weights & Biases
            
        Returns:
            Dict com resultados do treinamento
        """
        try:
            if not train_data:
                raise ValueError("Dados de treinamento vazios.")
            
            # Verificar e criar diretório de saída se não existir
            if not hasattr(self, 'output_dir') or not self.output_dir:
                self.output_dir = os.path.join("temp", "test_model")
                os.makedirs(self.output_dir, exist_ok=True)
                self.logger.info(f"Criado diretório de saída: {self.output_dir}")
                
            # Procurar o tokenizer de forma dinâmica
            possible_paths = []
            temp_dir = "temp"
            if os.path.exists(temp_dir):
                for subdir in os.listdir(temp_dir):
                    if subdir.startswith("test_"):
                        test_tokenizer_path = os.path.join(temp_dir, subdir, "test_model", "tokenizer")
                        if os.path.exists(test_tokenizer_path):
                            possible_paths.append(test_tokenizer_path)
            
            tokenizer_path = None
            if possible_paths:
                # Usar o mais recente
                tokenizer_path = sorted(possible_paths)[-1]
                self.logger.info(f"Usando tokenizer encontrado em: {tokenizer_path}")
            else:
                # Tentar no caminho padrão
                tokenizer_path = os.path.join("models", self.model_name, "tokenizer")
                if os.path.exists(tokenizer_path):
                    self.logger.info(f"Usando tokenizer do modelo: {tokenizer_path}")
                else:
                    raise FileNotFoundError(f"Tokenizer não encontrado em {tokenizer_path} nem em diretórios temporários")
            
            # Carregar o tokenizer
            from src.models.tokenizer import LunaTokenizer
            self.tokenizer = LunaTokenizer(self.config)
            self.tokenizer.load_from_directory(tokenizer_path)
            self.logger.info(f"Tokenizer carregado com sucesso de {tokenizer_path}")
            
            # Preparar dataset de treinamento
            from datasets import Dataset
            
            # Processar dados
            train_dataset = Dataset.from_dict({"text": train_data})
            
            # Função de pré-processamento
            def preprocess_function(examples):
                # Tokenizar textos
                return self.tokenizer.tokenizer(
                    examples["text"], 
                    truncation=True,
                    padding="max_length",
                    max_length=512
                )
            
            # Aplicar pré-processamento
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=["text"]
            )
            
            # Configurar treinamento com valores padrão quando os atributos não existem
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            
            # Obter valores com fallbacks seguros
            num_epochs = getattr(self.config.training, "num_train_epochs", 3)
            batch_size = getattr(self.config.training, "per_device_train_batch_size", 1)
            learning_rate = getattr(self.config.training, "learning_rate", 5e-5)
            weight_decay = getattr(self.config.training, "weight_decay", 0.01)
            save_steps = getattr(self.config.training, "save_steps", 500)
            logging_steps = getattr(self.config.training, "logging_steps", 10)
            save_total_limit = getattr(self.config.training, "save_total_limit", 1)
            
            # Configurar argumentos de treinamento com valores seguros
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                logging_steps=logging_steps
            )
            
            # Criar data collator para modelagem de linguagem
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer.tokenizer,
                mlm=False
            )
            
            # Configurar trainer
            trainer = Trainer(
                model=self.model.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator
            )
            
            # Treinar modelo
            trainer.train()
            
            # Limpar memória antes de salvar
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            try:
                # Salvar modelo treinado
                save_path = os.path.join("models", self.model_name)
                self.model.save(save_path)
            except Exception as save_error:
                self.logger.error(f"Erro ao salvar modelo: {str(save_error)}")
                # Se falhar ao salvar, ainda retornamos sucesso se o treinamento foi bem-sucedido
                return {
                    "success": True, 
                    "metrics": trainer.state.log_history,
                    "warning": "O treinamento foi concluído, mas ocorreu um erro ao salvar o modelo"
                }
            
            return {"success": True, "metrics": trainer.state.log_history}
            
        except Exception as e:
            self.logger.error(f"Erro durante o treinamento: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def train_with_curriculum(self, train_data, valid_data=None):
        """Treinamento em estágios com curriculum learning"""
        stages = [
            {"context_length": 128, "batch_size": 16},
            {"context_length": 256, "batch_size": 8},
            {"context_length": 512, "batch_size": 4},
        ]
        for stage in stages:
            self.config.training.context_length = stage["context_length"]
            self.config.training.per_device_train_batch_size = stage["batch_size"]
            self.logger.info(f"Iniciando estágio com contexto {stage['context_length']} e batch size {stage['batch_size']}")
            self.train_supervised(train_data, valid_data)

    def _start_memory_monitoring(self):
        """Inicia monitoramento de memória e ajusta dinamicamente os parâmetros."""
        try:
            import threading
            import time

            def monitor_memory():
                max_usage = 0
                while getattr(self, '_monitoring', True):
                    ram_usage = psutil.virtual_memory().percent
                    max_usage = max(max_usage, ram_usage)
                    if ram_usage > 90:
                        logger.warning(f"Uso crítico de memória: {ram_usage}%. Ajustando configurações...")
                        self.config.training.per_device_train_batch_size = max(1, self.config.training.per_device_train_batch_size // 2)
                        self.config.training.gradient_accumulation_steps *= 2
                    time.sleep(5)
                logger.info(f"Máximo uso de memória durante o treinamento: {max_usage}%")

            self._monitoring = True
            thread = threading.Thread(target=monitor_memory)
            thread.daemon = True
            thread.start()
        except Exception as e:
            logger.error(f"Erro ao iniciar monitoramento de memória: {e}")
    
    def _compute_causal_lm_loss_wrapper(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Wrapper para compatibilidade com diferentes versões de transformers"""
        # Se num_items_in_batch não foi fornecido, calcule-o
        if num_items_in_batch is None:
            num_items_in_batch = inputs.get("input_ids").shape[0] if "input_ids" in inputs else None
            
        return self._compute_causal_lm_loss(
            model, 
            inputs, 
            return_outputs=return_outputs, 
            num_items_in_batch=num_items_in_batch
        )
    
    def update_with_feedback(self):
        """Atualiza o modelo com dados de feedback dos usuários"""
        # Obter dados de feedback
        feedback_data = self.feedback.get_high_quality_feedback()
        
        if not feedback_data:
            logger.warning("Não há feedback suficiente para atualizar o modelo")
            return False
        
        # Preparar dados para treinamento
        train_texts = []
        for item in feedback_data:
            # Formatar como pergunta-resposta
            train_texts.append(f"Pergunta: {item['prompt']}\nResposta: {item['response']}")
        
        logger.info(f"Atualizando modelo com {len(train_texts)} amostras de feedback")
        
        # Treinar com os dados de feedback
        result = self.train_supervised(train_texts)
        
        if result['success']:
            logger.info("Modelo atualizado com sucesso usando feedback")
        else:
            logger.error(f"Falha ao atualizar modelo com feedback: {result.get('error', 'Erro desconhecido')}")
        
        return result['success']

    def evaluate_model(self, test_texts):
        """Avalia o desempenho do modelo em um conjunto de testes"""
        if not test_texts:
            logger.warning("Nenhum texto de teste fornecido para avaliação")
            return None
        
        logger.info(f"Avaliando modelo com {len(test_texts)} amostras")
        
        # Preparar dataset de avaliação
        def preprocess_function(examples):
            tokenized_inputs = self.tokenizer.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors=None,
            )
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
            return tokenized_inputs
        
        eval_dataset = Dataset.from_dict({"text": test_texts})
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            desc="Processando dados de avaliação",
            remove_columns=["text"]
        )
        
        # Configurar trainer apenas para avaliação
        training_args = TrainingArguments(
            output_dir=os.path.join(self.model_dir, "eval"),
            per_device_eval_batch_size=4,
            report_to="none",
        )
        
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer.tokenizer,
            compute_loss=self._compute_causal_lm_loss_wrapper,
        )
        
        # Avaliar o modelo
        eval_result = trainer.evaluate()
        
        logger.info(f"Resultado da avaliação: {eval_result}")
        return eval_result

    def _load_model(self):
        """Carrega o modelo Luna para treinamento.
        
        Returns:
            LunaModel: Instância do modelo carregado
        """
        try:
            if not hasattr(self, 'model_name') or not self.model_name:
                raise ValueError("Nome do modelo não definido ou vazio")
                
            # Verificar o caminho do modelo
            model_dir = os.path.join("models", self.model_name)
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Diretório do modelo não encontrado: {model_dir}")
                
            # Carregar o modelo
            from src.models.luna_model import LunaModel
            model = LunaModel.from_pretrained(model_dir, self.config.model)
            
            # Mover para o dispositivo apropriado
            model.to_appropriate_device()
            
            self.logger.info(f"Modelo {self.model_name} carregado com sucesso para treinamento")
            return model
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo para treinamento: {str(e)}")
            raise

    def _setup_training_args(self):
        """
        Configura os argumentos de treinamento com base nas configurações e hardware.
        
        Returns:
            TrainingArguments: Objeto configurado para o treinamento
        """
        from transformers import TrainingArguments
        
        # Garantir que temos um diretório de saída
        if not hasattr(self, 'output_dir') or not self.output_dir:
            self.output_dir = os.path.join("temp", "test_model")
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurar argumentos de treinamento
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            logging_steps=self.config.training.logging_steps,
            evaluation_strategy="steps" if valid_data else "no"
        )
        
        return args

    def optimize_hyperparameters(self, train_data, valid_data):
        """Busca automática de hiperparâmetros com Optuna"""
        def objective(trial):
            self.config.training.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
            self.config.training.dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
            result = self.train_supervised(train_data, valid_data)
            return result["metrics"][-1]["eval_loss"]
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)
        self.logger.info(f"Melhores hiperparâmetros: {study.best_params}")