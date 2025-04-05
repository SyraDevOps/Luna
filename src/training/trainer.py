import os
import logging
import traceback
import torch
from typing import List, Optional, Dict, Union, Tuple, Any
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import transformers
import numpy as np
from tqdm import tqdm

from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer
from src.models.feedback_system import FeedbackSystem
from src.config.config import Config
from src.utils.callbacks import CustomTrainingCallback
from src.utils.hardware_utils import detect_hardware, setup_memory_efficient_training
import gc
import psutil

logger = logging.getLogger(__name__)

class LunaTrainer:
    """Classe de treinamento para modelos Luna"""
    
    def __init__(self, model_path, config):
        """Inicializa o trainer."""
        self.config = config
        self.device = detect_hardware()
        self.logger = logging.getLogger(__name__)
        
        # Tratar tanto nomes de modelo quanto caminhos completos
        if os.path.isabs(model_path) or os.path.exists(model_path):
            self.model_dir = model_path
        else:
            self.model_dir = os.path.join("models", model_path)
        
        self.output_dir = os.path.join(self.model_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ajustar hiperparâmetros com base no hardware
        self._adjust_hyperparams_for_hardware()
        
        # Configurar modelo
        self.logger.info(f"Carregando modelo: {self.model_dir}")
        self.model = LunaModel.from_pretrained(
            self.model_dir,
            config=self.config.model,
            device_setup={
                "low_cpu_mem_usage": True,
                "use_fp16": self.device.gpu_available,
            }
        )
        
        # Carregar tokenizer
        self.tokenizer = LunaTokenizer(self.config)
        self.tokenizer.load(os.path.join(self.model_dir, "tokenizer"))
    
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
        """Ajusta a configuração de treinamento com base no hardware"""
        # Ajustar batch size
        optimal_batch_size = self.hardware_profile.get_optimal_batch_size()
        
        # Atualizar configuração de treino
        if hasattr(self.config.training, 'per_device_train_batch_size'):
            if self.config.training.per_device_train_batch_size > optimal_batch_size:
                logger.info(f"Ajustando batch size de {self.config.training.per_device_train_batch_size} para {optimal_batch_size}")
                self.config.training.per_device_train_batch_size = optimal_batch_size
                self.config.training.per_device_eval_batch_size = optimal_batch_size
        
        # Ajustar gradient accumulation para compensar batch size menor
        if self.hardware_profile.system_type == "low-end" and hasattr(self.config.training, 'gradient_accumulation_steps'):
            # Aumentar passos de acumulação para compensar batch size menor
            self.config.training.gradient_accumulation_steps = max(4, self.config.training.gradient_accumulation_steps)
            logger.info(f"Ajustando gradient accumulation steps para {self.config.training.gradient_accumulation_steps}")
        
        # Configurar uso de memória mista
        if self.hardware_profile.has_cuda and self.hardware_profile.system_type != "low-end":
            self.config.training.fp16 = True
            logger.info("Habilitando treinamento FP16 para GPU")
    
    def _compute_causal_lm_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Função personalizada para calcular a perda de modelagem causal de linguagem."""
        try:
            # Validar inputs
            if "input_ids" not in inputs or inputs["input_ids"] is None:
                raise ValueError("Os dados de entrada não contêm 'input_ids' válidos.")
            
            # Garantir que labels estão presentes
            inputs["labels"] = inputs.get("labels", inputs["input_ids"].clone())

            # Ajustar position_ids
            seq_length = inputs["input_ids"].shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs["input_ids"].device)
            position_ids = position_ids.unsqueeze(0).expand(inputs["input_ids"].shape[0], -1)
            inputs["position_ids"] = torch.clamp(position_ids, 0, model.config.n_positions - 1)

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss or torch.tensor(1.0, requires_grad=True, device=inputs["input_ids"].device)
            return (loss, outputs) if return_outputs else loss
        except Exception as e:
            self.logger.error(f"Erro em _compute_causal_lm_loss: {str(e)}")
            return torch.tensor(1.0, requires_grad=True, device="cpu")
    
    def train_supervised(self, training_data, validation_data=None, **kwargs):
        """Treina o modelo com aprendizado supervisionado."""
        try:
            # Validar dados de treinamento
            if not training_data or len(training_data) == 0:
                raise ValueError("Os dados de treinamento estão vazios ou inválidos.")
            
            # Configuração inicial
            self.logger.info(f"Treinando no dispositivo: {self.device}")
            
            # Garantir pelo menos 5 épocas
            if getattr(self.config.training, 'num_train_epochs', 0) < 5:
                self.logger.info("Definindo número mínimo de épocas para 5")
                self.config.training.num_train_epochs = 5
            
            # Verificar e, se necessário, treinar o tokenizer
            if not hasattr(self.tokenizer, 'tokenizer') or self.tokenizer.tokenizer is None:
                self.logger.warning("Tokenizer não disponível, treinando um novo...")
                tokenizer_dir = os.path.join(self.model_dir, "tokenizer")
                os.makedirs(tokenizer_dir, exist_ok=True)
                self.tokenizer.train_and_save(training_data, tokenizer_dir)
                
            # Verificar novamente se o tokenizer está disponível
            if not hasattr(self.tokenizer, 'tokenizer') or self.tokenizer.tokenizer is None:
                self.logger.error("Tokenizer não inicializado corretamente")
                raise ValueError("Tokenizer não está disponível. Verifique a inicialização.")
                
            # Função de tokenização segura
            def tokenize_function(examples):
                return self.tokenizer.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors=None,
                )
            
            # Configurar argumentos de treinamento
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.config.training.num_train_epochs,
                per_device_train_batch_size=self.config.training.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
                logging_dir=os.path.join(self.output_dir, "logs"),
                logging_steps=self.config.training.logging_steps,
                evaluation_strategy="epoch" if validation_data else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if validation_data else False,
                save_total_limit=2,
                report_to="wandb" if getattr(self.config.training, 'use_wandb', False) else "none",
            )
            
            # Preparar datasets
            train_dataset = Dataset.from_dict({"text": training_data})
            try:
                train_dataset = train_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["text"]
                )
            except Exception as e:
                self.logger.error(f"Erro ao tokenizar dados de treinamento: {str(e)}")
                raise
            
            eval_dataset = None
            if validation_data:
                eval_dataset = Dataset.from_dict({"text": validation_data})
                try:
                    eval_dataset = eval_dataset.map(
                        tokenize_function,
                        batched=True,
                        remove_columns=["text"]
                    )
                except Exception as e:
                    self.logger.error(f"Erro ao tokenizar dados de validação: {str(e)}")
                    raise
            
            # Criar o trainer
            trainer = Trainer(
                model=self.model.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer.tokenizer,
            )
            
            # Iniciar treinamento
            self.logger.info(f"Iniciando treinamento por {self.config.training.num_train_epochs} épocas...")
            trainer_output = trainer.train()
            
            # Salvar modelo
            self.logger.info("Salvando modelo treinado...")
            self.model.save(self.model_dir)
            
            return {"success": True, "metrics": trainer.state.log_history}
        
        except Exception as e:
            self.logger.error(f"Erro durante o treinamento: {str(e)}")
            return {"success": False, "error": str(e)}
    
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
    
    def _compute_causal_lm_loss_wrapper(self, model, inputs, return_outputs=False):
        """Wrapper para compatibilidade com diferentes versões de transformers"""
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