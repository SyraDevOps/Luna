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
import shutil
import json

from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer
from src.models.feedback_system import FeedbackSystem
from src.config.config import Config
from src.utils.callbacks import CustomTrainingCallback
from src.utils.hardware_utils import detect_hardware, setup_memory_efficient_training
from src.models.adaptive_tokenizer import AdaptiveTokenizer
import gc
import psutil
import tempfile
from src.utils.wandb_utils import initialize_wandb, is_wandb_available
from src.optimization.automl_hyperparams import LunaAutoML, DynamicHyperparamOptimizer

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
        self.device = detect_hardware()
        
        # Configurar diretório temporário para arquivos de treinamento
        self.output_dir = os.path.join("temp", "training_" + model_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Diretório temporário configurado em: {self.output_dir}")
        
        # Diretório do modelo
        self.model_dir = os.path.join("models", model_name)
        
        # Inicializar parâmetros de treinamento
        self.gradient_accumulation_steps = getattr(config.training, "gradient_accumulation_steps", 1)
        
        # Ajustar parâmetros de treinamento com base no hardware
        self._adjust_hyperparams_for_hardware()
        
        # Carregar o modelo
        try:
            self.model = self._load_model()
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo para treinamento: {str(e)}")
            raise
        
        # Inicializar sistema de feedback se necessário
        self.feedback = FeedbackSystem(self.model_name, config)
        
        # Inicializar tokenizer adaptativo se configurado
        self.tokens_learning_during_training = getattr(config, "tokens_learning_during_training", False)
        self.collect_training_tokens = getattr(config, "collect_training_tokens", False)
        
        if self.tokens_learning_during_training or self.collect_training_tokens:
            self.adaptive_tokenizer = AdaptiveTokenizer(model_name, config, self.tokenizer.tokenizer)
            self.tokens_processed = 0
            self.tokens_check_frequency = 1000  # Verificar a cada 1000 tokens processados
        
        # Iniciar monitoramento de memória
        self._start_memory_monitoring()
    
    def _adjust_hyperparams_for_hardware(self):
        """Ajusta hiperparâmetros com base no hardware disponível."""
        # Verificar se os atributos existem para evitar AttributeError
        gpu_available = getattr(self.device, 'gpu_available', False)
        gpu_memory_gb = getattr(self.device, 'gpu_memory_gb', 0)
        
        # Ambiente Colab - configurações conservadoras
        is_colab = os.environ.get('COLAB_GPU', '') == '1'
        if is_colab:
            logger.info("Ambiente Google Colab detectado. Usando configurações otimizadas.")
            self.batch_size = 4
            self.gradient_accumulation_steps = 4
            self.optimizer_parameters['lr'] = 1e-4
            return
            
        # Resto da lógica existente usando verificações seguras de atributos
        if gpu_available and gpu_memory_gb > 8:
            # Hardware potente - configuração original
            pass  # manter configurações padrão
        elif gpu_available:
            # GPU com memória limitada
            self.batch_size = max(2, self.batch_size // 2)
            self.gradient_accumulation_steps *= 2
        else:
            # Apenas CPU
            self.batch_size = 1
            self.gradient_accumulation_steps *= 4
    
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
            return torch.tensor(1.0, requires_grad=True, device=inputs["input_ids"].device)
    
    def _process_batch_for_tokens(self, texts):
        """Processa um batch de textos para detecção de tokens"""
        if not (self.tokens_learning_during_training or self.collect_training_tokens):
            return
            
        for text in texts:
            self.adaptive_tokenizer.analyze_text(text)
            self.tokens_processed += len(text.split())
            
            # Verificar se é hora de atualizar o tokenizer
            if self.tokens_learning_during_training and self.tokens_processed >= self.tokens_check_frequency:
                self.logger.info("Verificando novos tokens durante treinamento...")
                num_added = self.adaptive_tokenizer.extend_tokenizer(self.model)
                if num_added > 0:
                    self.logger.info(f"Adicionados {num_added} novos tokens ao vocabulário durante treinamento")
                self.tokens_processed = 0
    
    def _save_token_candidates(self):
        """Salva tokens candidatos coletados durante o treinamento"""
        if not self.collect_training_tokens:
            return
            
        try:
            # Obter candidatos com suas frequências
            candidates = {term: count for term, count in 
                         self.adaptive_tokenizer.unknown_terms_counter.most_common(10000)}
            
            # Salvar no arquivo de coleta
            with open(self.adaptive_tokenizer.token_collection_file, 'w', encoding='utf-8') as f:
                json.dump(candidates, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Salvos {len(candidates)} candidatos a tokens em {self.adaptive_tokenizer.token_collection_file}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar candidatos a tokens: {str(e)}")
    
    def train_supervised(self, train_data, valid_data=None, use_wandb=False, num_train_epochs=None, output_dir=None, dynamic_hp_opt=False):
        """Treina o modelo usando aprendizado supervisionado"""
        try:
            # Import wandb at the beginning to make it available in all code paths
            if use_wandb:
                import wandb
                use_wandb = initialize_wandb(
                    config=vars(self.config),
                    run_name=f"train_{self.model_name}",
                    project_name="lunagpt"
                )
            
            if not train_data:
                raise ValueError("Dados de treinamento vazios.")
                
            # Encontrar e carregar tokenizer
            tokenizer_path = os.path.join(self.model_dir, "tokenizer")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer não encontrado em {tokenizer_path}")
            
            self.tokenizer = LunaTokenizer(self.config)
            self.tokenizer.load_from_directory(tokenizer_path)
            self.logger.info(f"Tokenizer carregado com sucesso de {tokenizer_path}")
            
            # Processar dados de treinamento para tokens
            if self.tokens_learning_during_training or self.collect_training_tokens:
                self.logger.info("Analisando dados de treinamento para possíveis novos tokens...")
                for text in tqdm(train_data, desc="Analisando tokens"):
                    self._process_batch_for_tokens([text])
            
            # Preparar dataset de treinamento
            train_dataset = Dataset.from_dict({"text": train_data})
            
            # Função de pré-processamento
            def preprocess_function(examples):
                return self.tokenizer.tokenizer(
                    examples["text"], 
                    truncation=True,
                    padding="max_length",
                    max_length=getattr(self.config.training, "context_length", 512)
                )
            
            # Aplicar pré-processamento
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=["text"],
                desc="Processando dados de treinamento"
            )
            
            # Preparar dataset de validação se fornecido
            eval_dataset = None
            if valid_data and len(valid_data) > 0:
                eval_dataset = Dataset.from_dict({"text": valid_data})
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    remove_columns=["text"],
                    desc="Processando dados de validação"
                )
                self.logger.info(f"Dataset de validação com {len(valid_data)} exemplos preparado")
            
            # Obter valores com fallbacks seguros
            n_epochs = num_train_epochs if num_train_epochs is not None else getattr(self.config.training, "num_train_epochs", 3)
            batch_size = getattr(self.config.training, "per_device_train_batch_size", 1)
            learning_rate = getattr(self.config.training, "learning_rate", 5e-5)
            weight_decay = getattr(self.config.training, "weight_decay", 0.01)
            save_steps = getattr(self.config.training, "save_steps", 500)
            logging_steps = getattr(self.config.training, "logging_steps", 10)
            save_total_limit = getattr(self.config.training, "save_total_limit", 1)
            gradient_accumulation = getattr(self.config.training, "gradient_accumulation_steps", 1)
            
            # Configurar argumentos de treinamento - removing problematic parameters
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=n_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                logging_steps=logging_steps,
                gradient_accumulation_steps=gradient_accumulation,
                report_to="wandb" if use_wandb else "none",
                push_to_hub=False
            )
            
            # Handle evaluation separately if it exists
            if eval_dataset:
                # Try to set evaluation frequency based on version compatibility
                try:
                    training_args.eval_steps = save_steps
                    training_args.evaluation_strategy = "steps"
                    training_args.load_best_model_at_end = True
                    training_args.metric_for_best_model = "loss"  # Use loss as the metric for selecting best model
                    training_args.greater_is_better = False  # For loss, lower is better
                except AttributeError:
                    self.logger.warning("Evaluation attributes not supported in this version. Evaluation may not work as expected.")
            
            # Criar data collator para modelagem de linguagem
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer.tokenizer,
                mlm=False
            )
            
            # Adicionar callbacks
            callbacks = [CustomTrainingCallback(timeout_min=120, monitor_memory=True)]
            
            # Configurar trainer
            trainer = Trainer(
                model=self.model.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
                compute_metrics=self._compute_metrics if eval_dataset else None
            )
            
            # Treinar modelo
            self.logger.info(f"Iniciando treinamento por {n_epochs} épocas com batch size {batch_size}")
            trainer.train()
            
            # Avaliar modelo se houver dados de validação
            eval_results = {}
            if eval_dataset:
                self.logger.info("Avaliando modelo em dados de validação")
                eval_results = trainer.evaluate()
                self.logger.info(f"Resultados da avaliação: {eval_results}")
            
            # Limpar memória antes de salvar
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            try:
                # Salvar modelo treinado como arquivo .pt
                self._save_model_as_pt(trainer.model)
                
                # Salvar no Weights & Biases se ativado
                if use_wandb:
                    self._save_to_wandb()
                    
            except Exception as save_error:
                self.logger.error(f"Erro ao salvar modelo: {str(save_error)}")
                return {
                    "success": True,
                    "metrics": trainer.state.log_history,
                    "warning": "O treinamento foi concluído, mas ocorreu um erro ao salvar o modelo",
                    "eval_results": eval_results
                }
            
            # Salvar candidatos a tokens ao final do treinamento
            if self.collect_training_tokens:
                self._save_token_candidates()
            
            # Final do treinamento: finalizar wandb se estiver ativo
            if use_wandb and is_wandb_available():
                import wandb
                if wandb.run is not None:
                    wandb.finish()
                
            return {"success": True, "metrics": trainer.state.log_history, "eval_results": eval_results}
            
        except Exception as e:
            self.logger.error(f"Erro durante o treinamento: {str(e)}")
            traceback.print_exc()
            if use_wandb:
                import wandb  # Import again to be safe
                if 'wandb' in globals() and wandb.run is not None:
                    wandb.finish()
            return {"success": False, "error": str(e)}
    
    def _save_model_as_pt(self, trained_model):
        """Salva o modelo treinado como arquivo .pt"""
        # Garantir que o diretório do modelo existe
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Definir caminho para o arquivo .pt
        pt_file_path = os.path.join(self.model_dir, "luna_model.pt")
        
        # Salvar o modelo treinado
        torch.save(trained_model.state_dict(), pt_file_path)
        self.logger.info(f"Modelo salvo como arquivo .pt em: {pt_file_path}")
        
        # Salvar configuração do modelo
        config_path = os.path.join(self.model_dir, "config.json")
        trained_model.config.save_pretrained(self.model_dir)
        self.logger.info(f"Configuração do modelo salva em: {config_path}")
        
        # Atualizar o modelo da instância com o modelo treinado
        self.model.model = trained_model
        
        # Salvar componentes adicionais
        components_dir = os.path.join(self.model_dir, "components")
        os.makedirs(components_dir, exist_ok=True)
        
        # Salvar MoE blocks se existirem
        if hasattr(self.model, "moe_blocks") and self.model.moe_blocks is not None:
            torch.save(self.model.moe_blocks, os.path.join(components_dir, "moe.pt"))
            
        # Salvar hypernet se existir
        if hasattr(self.model, "hypernet") and self.model.hypernet is not None:
            torch.save(self.model.hypernet, os.path.join(components_dir, "hypernet.pt"))
            
        # Salvar growing network se existir
        if hasattr(self.model, "growing_network") and self.model.growing_network is not None:
            torch.save(self.model.growing_network, os.path.join(components_dir, "growing_network.pt"))
            
        # Salvar o modelo completo em um único arquivo unificado (inclui tudo)
        full_model_path = os.path.join(self.model_dir, "full_model.pt")
        torch.save(self.model, full_model_path)
        self.logger.info(f"Modelo completo salvo em: {full_model_path}")
        
        return pt_file_path
    
    def _save_to_wandb(self):
        """Salva o modelo no Weights & Biases"""
        if not is_wandb_available():
            self.logger.warning("Weights & Biases não está instalado.")
            return
            
        import wandb
        if wandb.run is None:
            self.logger.warning("Nenhuma execução ativa do Weights & Biases encontrada.")
            return
        
        try:
            # Salvar artefatos no W&B
            model_artifact = wandb.Artifact(
                name=f"model-{self.model_name}",
                type="model", 
                description=f"Modelo LunaGPT {self.model_name}"
            )
            
            # Adicionar diretório completo do modelo
            model_artifact.add_dir(self.model_dir)
            
            # Registrar artefato
            wandb.log_artifact(model_artifact)
            self.logger.info(f"Modelo salvo no Weights & Biases como artefato: model-{self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar no Weights & Biases: {str(e)}")
    
    def _compute_metrics(self, eval_pred):
        """Calcula métricas de avaliação personalizadas"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calcular perplexidade
        loss = self._compute_causal_lm_loss(self.model.model, {
            "input_ids": torch.tensor(labels, device=self.model.model.device),
            "labels": torch.tensor(labels, device=self.model.model.device)
        }).item()
        
        perplexity = np.exp(loss)
        
        return {
            "perplexity": perplexity,
            "loss": loss
        }
    
    def train_with_curriculum(self, train_data, valid_data=None, use_wandb=False):
        """Treinamento em estágios com curriculum learning"""
        
        # Verificar se o usuário definiu um número personalizado de épocas
        custom_epochs = getattr(self.config.training, "num_train_epochs", None)
        
        # Definir estágios padrão
        stages = [
            {"context_length": 128, "batch_size": 16, "epochs": 1},
            {"context_length": 256, "batch_size": 8, "epochs": 2},
            {"context_length": 512, "batch_size": 4, "epochs": 2},
        ]
        
        # Se houver épocas personalizadas, ajustar proporcionalmente
        if custom_epochs is not None and custom_epochs != 5:  # 5 é a soma padrão de épocas
            # Calcular fator de escala
            scale_factor = custom_epochs / 5
            # Ajustar épocas de cada estágio
            for stage in stages:
                stage["epochs"] = max(1, round(stage["epochs"] * scale_factor))
            self.logger.info(f"Ajustando estágios do curriculum para total de {custom_epochs} épocas")
        
        # Ajustar estágios com base no hardware
        if self.device.system_type == "low-end":
            stages = [
                {"context_length": 64, "batch_size": 2, "epochs": 1},
                {"context_length": 128, "batch_size": 1, "epochs": 2},
                {"context_length": 256, "batch_size": 1, "epochs": 2},
            ]
        
        for i, stage in enumerate(stages):
            self.logger.info(f"Iniciando estágio {i+1}/{len(stages)} do curriculum learning")
            self.logger.info(f"Contexto: {stage['context_length']}, Batch size: {stage['batch_size']}, Épocas: {stage['epochs']}")
            
            self.config.training.context_length = stage["context_length"]
            self.config.training.per_device_train_batch_size = stage["batch_size"]
            
            # Se for o último estágio, ativar wandb se solicitado
            use_wandb_stage = use_wandb and (i == len(stages) - 1)
            
            result = self.train_supervised(
                train_data, 
                valid_data, 
                use_wandb=use_wandb_stage,
                num_train_epochs=stage["epochs"]
            )
            
            if not result["success"]:
                self.logger.error(f"Falha no estágio {i+1}: {result.get('error', 'Erro desconhecido')}")
                return result
        
        self.logger.info("Curriculum learning concluído com sucesso!")
        return {"success": True, "message": "Treinamento com curriculum learning concluído"}

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
    
    def update_with_feedback(self, use_wandb=False):
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
        
        # Treinar com os dados de feedback (menos épocas para refinamento)
        result = self.train_supervised(train_texts, use_wandb=use_wandb, num_train_epochs=1)
        
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
        
        # Configurar trainer para avaliação
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
        """Carrega o modelo Luna para treinamento, priorizando arquivo .pt"""
        try:
            if not hasattr(self, 'model_name') or not self.model_name:
                raise ValueError("Nome do modelo não definido ou vazio")
                
            # Verificar o caminho do modelo
            model_dir = os.path.join("models", self.model_name)
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Diretório do modelo não encontrado: {model_dir}")
            
            # Importar e registrar a classe como segura para carregamento
            from src.models.luna_model import LunaModel
            from src.models.moe import MoEBlock
            from src.models.growing_network import StateSpaceLayer, GrowingNetwork
            from src.models.hypernet import HyperNetwork
            
            # Adicionar classes seguras para desserialização
            import torch.serialization
            torch.serialization.add_safe_globals([LunaModel, MoEBlock, StateSpaceLayer, 
                                                 GrowingNetwork, HyperNetwork])
            
            # Verificar se existe arquivo .pt unificado
            full_model_path = os.path.join(model_dir, "full_model.pt")
            if os.path.exists(full_model_path):
                self.logger.info(f"Carregando modelo completo de {full_model_path}")
                # Usar weights_only=False para classes personalizadas
                model = torch.load(full_model_path, weights_only=False)
                model.to_appropriate_device()
                return model
                
            # Caso não exista arquivo unificado, criar um novo modelo
            self.logger.info(f"Criando nova instância do modelo {self.model_name}")
            model = LunaModel.from_scratch(self.config.model)
            model.to_appropriate_device()
            return model
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo para treinamento: {str(e)}")
            raise
    
    def train_with_automl(
        self, 
        train_data, 
        valid_data=None, 
        automl_config=None, 
        use_wandb=False, 
        num_train_epochs=None,
        base_output_dir=None
    ):
        """
        Treina o modelo usando otimização automática de hiperparâmetros
        
        Args:
            train_data: Dados de treinamento
            valid_data: Dados de validação
            automl_config: Configuração do AutoML (ou None para usar padrão)
            use_wandb: Se deve usar o wandb para tracking
            num_train_epochs: Número de épocas (por trial)
            base_output_dir: Diretório base para salvar resultados
        
        Returns:
            dict: Resultados do melhor modelo treinado
        """
        self.logger.info("Iniciando treinamento com AutoML...")
        
        # Configurar diretório para resultados do AutoML
        if base_output_dir is None:
            base_output_dir = os.path.join(self.model_dir, "automl")
        
        os.makedirs(base_output_dir, exist_ok=True)
        storage_path = os.path.join(base_output_dir, "automl_storage.db")
        
        # Configurar AutoML
        automl_config = automl_config or {}
        automl = LunaAutoML(
            study_name=automl_config.get("study_name", f"lunagpt_{self.model_name}"),
            storage_path=storage_path,
            metric=automl_config.get("metric", "validation_loss"),
            direction=automl_config.get("direction", "minimize"),
            n_trials=automl_config.get("n_trials", 10),
            n_jobs=automl_config.get("n_jobs", 1),
            pruner_type=automl_config.get("pruner_type", "hyperband"),
        )
        
        # Definir função de treinamento para o AutoML
        def train_trial_fn(
            learning_rate=5e-5,
            weight_decay=0.01,
            batch_size=4,
            gradient_accumulation_steps=1,
            warmup_ratio=0.1,
            dropout_rate=0.1,
            attention_dropout=0.1,
            **extra_args
        ):
            # Atualizar configurações com os hiperparâmetros amostrados
            import copy
            config_copy = copy.deepcopy(self.config)
            
            # Atualizar configurações de treinamento
            config_copy.training.learning_rate = learning_rate
            config_copy.training.weight_decay = weight_decay
            config_copy.training.per_device_train_batch_size = batch_size
            config_copy.training.gradient_accumulation_steps = gradient_accumulation_steps
            config_copy.training.warmup_ratio = warmup_ratio
            
            # Atualizar configurações do modelo
            if hasattr(config_copy.model, "dropout_rate"):
                config_copy.model.dropout_rate = dropout_rate
            if hasattr(config_copy.model, "attention_dropout"):
                config_copy.model.attention_dropout = attention_dropout
                
            # Aplicar configurações extras
            for key, value in extra_args.items():
                parts = key.split('.')
                if len(parts) == 2:
                    section, name = parts
                    if hasattr(config_copy, section) and hasattr(getattr(config_copy, section), name):
                        setattr(getattr(config_copy, section), name, value)
            
            # Criar novo trainer com estas configurações
            trial_trainer = LunaTrainer(self.model_name, config_copy)
            
            # Definir caminho de saída para o trial
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trial_dir = os.path.join(base_output_dir, f"trial_{timestamp}")
            os.makedirs(trial_dir, exist_ok=True)
            
            # Treinar modelo com configurações atualizadas
            results = trial_trainer.train_supervised(
                train_data, 
                valid_data,
                use_wandb=use_wandb,
                num_train_epochs=num_train_epochs or self.config.training.num_train_epochs,
                output_dir=trial_dir
            )
            
            # Retornar métrica de avaliação
            metric_value = results.get(automl.metric, float('inf'))
            return metric_value
        
        # Executar otimização
        best_params = automl.optimize(train_trial_fn)
        
        # Treinar modelo final com os melhores hiperparâmetros
        self.logger.info(f"Treinando modelo final com os melhores hiperparâmetros: {best_params}")
        final_results = train_trial_fn(**best_params)
        
        # Salvar os melhores hiperparâmetros
        best_params_path = os.path.join(base_output_dir, "best_hyperparams.json")
        with open(best_params_path, 'w') as f:
            import json
            json.dump(best_params, f, indent=2)
        
        self.logger.info(f"AutoML concluído. Melhores hiperparâmetros salvos em: {best_params_path}")
        return final_results