import os
import time
import logging
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Union, Tuple, Any
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer
from src.models.feedback_system import FeedbackSystem
from src.models.supervised_dataset import SupervisedDataset
from src.config.config import Config
from src.utils.callbacks import CustomTrainingCallback
from src.utils.hardware_utils import detect_hardware, setup_memory_efficient_training
from src.models.adaptive_tokenizer import AdaptiveTokenizer
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
        self.training_batch_size = 2 if self.device.system_type == "low-end" else 4
        self.eval_batch_size = 1 if self.device.system_type == "low-end" else 2
        
        # Ajustar parâmetros de treinamento com base no hardware
        self._adjust_hyperparams_for_hardware()
        
        # Carregar o modelo
        try:
            self.model, self.tokenizer = self._load_model()
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
        
        # Inicializar sistema de feedback se necessário
        self.feedback = FeedbackSystem(config)
        
        # Inicializar tokenizer adaptativo se configurado
        self.tokens_learning_during_training = getattr(config, "tokens_learning_during_training", False)
        self.collect_training_tokens = getattr(config, "collect_training_tokens", False)
        
        if self.tokens_learning_during_training or self.collect_training_tokens:
            self.adaptive_tokenizer = AdaptiveTokenizer(model_name, config, self.tokenizer)
    
    def _adjust_hyperparams_for_hardware(self):
        """Ajusta hiperparâmetros baseado no hardware disponível"""
        if self.device.system_type == "low-end":
            self.config.training.per_device_train_batch_size = 1
            self.config.training.gradient_accumulation_steps = 4
            self.config.training.gradient_checkpointing = True
        elif self.device.system_type == "mid-range":
            self.config.training.per_device_train_batch_size = 2
            self.config.training.gradient_accumulation_steps = 2
        # high-end usa configurações padrão
    
    def _compute_causal_lm_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Calcula a perda para modelagem de linguagem causal"""
        try:
            outputs = model(**inputs)
            
            # Shift dos labels para causal LM
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            
            # Calcular perda
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
        except Exception as e:
            logger.error(f"Erro no cálculo de perda: {e}")
            # Retornar perda zero como fallback
            return torch.tensor(0.0, requires_grad=True)
    
    def _process_batch_for_tokens(self, texts):
        """Processa batch para coletar tokens candidatos"""
        if hasattr(self, 'adaptive_tokenizer'):
            for text in texts:
                self.adaptive_tokenizer.analyze_text(text)
    
    def _save_token_candidates(self):
        """Salva candidatos a tokens coletados durante treinamento"""
        if hasattr(self, 'adaptive_tokenizer') and self.adaptive_tokenizer.unknown_terms_counter:
            candidates_file = os.path.join(self.model_dir, "token_candidates.json")
            candidates_dict = dict(self.adaptive_tokenizer.unknown_terms_counter)
            
            try:
                import json
                with open(candidates_file, 'w', encoding='utf-8') as f:
                    json.dump(candidates_dict, f, ensure_ascii=False, indent=2)
                logger.info(f"Candidatos a tokens salvos em {candidates_file}")
            except Exception as e:
                logger.error(f"Erro ao salvar candidatos a tokens: {e}")
    
    def train_supervised(self, train_data, valid_data=None, use_wandb=False, num_train_epochs=None, output_dir=None, dynamic_hp_opt=False):
        """
        Treina o modelo com dados supervisionados
        """
        try:
            logger.info(f"Iniciando treinamento supervisionado com {len(train_data)} amostras")
            
            # Configurar memória eficiente
            setup_memory_efficient_training()
            
            # Processar dados para coleta de tokens se ativado
            if self.collect_training_tokens:
                self._process_batch_for_tokens(train_data)
            
            # Criar datasets
            train_dataset = SupervisedDataset(train_data, self.tokenizer, max_length=512)
            eval_dataset = None
            if valid_data:
                eval_dataset = SupervisedDataset(valid_data, self.tokenizer, max_length=512)
            
            # Configurar argumentos de treinamento
            training_args = TrainingArguments(
                output_dir=output_dir or self.output_dir,
                num_train_epochs=num_train_epochs or 1,
                per_device_train_batch_size=self.training_batch_size,
                per_device_eval_batch_size=self.eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=os.path.join(self.output_dir, "logs"),
                logging_steps=self.logging_steps,
                save_steps=self.save_steps,
                eval_steps=self.eval_steps if valid_data else None,
                save_total_limit=3,
                load_best_model_at_end=False,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=["wandb"] if use_wandb else [],
                dataloader_num_workers=0,
                gradient_checkpointing=self.use_gradient_checkpointing,
                fp16=self.use_fp16,
                push_to_hub=False,
                remove_unused_columns=False
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM, não masked LM
            )
            
            # Inicializar trainer
            trainer = Trainer(
                model=self.model.model if hasattr(self.model, 'model') else self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics if eval_dataset else None,
                callbacks=[CustomTrainingCallback(timeout_min=30)]
            )
            
            # Override do método compute_loss
            trainer.compute_loss = lambda model, inputs, return_outputs=False: self._compute_causal_lm_loss_wrapper(model, inputs, return_outputs)
            
            # Treinar
            trainer.train()
            
            # Salvar modelo treinado
            self._save_model_as_pt(trainer.model)
            
            # Salvar candidatos a tokens se coletados
            if self.collect_training_tokens:
                self._save_token_candidates()
            
            # Salvar no wandb se configurado
            if use_wandb:
                self._save_to_wandb()
            
            logger.info("Treinamento supervisionado concluído com sucesso")
            return {"success": True, "model_path": self.model_dir}
            
        except Exception as e:
            logger.error(f"Erro durante treinamento supervisionado: {e}")
            return {"success": False, "error": str(e)}
    
    def _save_model_as_pt(self, trained_model):
        """Salva o modelo treinado em formato .pt"""
        try:
            # Salvar em formato HuggingFace
            trained_model.save_pretrained(self.model_dir)
            
            # Salvar também em formato .pt
            pt_path = os.path.join(self.model_dir, "model.pt")
            torch.save(trained_model.state_dict(), pt_path)
            
            # Criar nome alternativo baseado no nome do modelo
            alt_path = os.path.join(self.model_dir, f"model_{self.model_name}.pt")
            torch.save(trained_model.state_dict(), alt_path)
            
            logger.info(f"Modelo salvo em {self.model_dir}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
    
    def _save_to_wandb(self):
        """Salva artefatos no wandb"""
        if is_wandb_available():
            try:
                import wandb
                artifact = wandb.Artifact(f"luna_model_{self.model_name}", type="model")
                artifact.add_dir(self.model_dir)
                wandb.log_artifact(artifact)
                logger.info("Modelo salvo no wandb")
            except Exception as e:
                logger.error(f"Erro ao salvar no wandb: {e}")
    
    def _compute_metrics(self, eval_pred):
        """Calcula métricas de avaliação"""
        predictions, labels = eval_pred
        
        # Calcular perplexidade
        try:
            # Reshape predictions para calcular perplexidade
            predictions = predictions.reshape(-1, predictions.shape[-1])
            labels = labels.reshape(-1)
            
            # Filtrar tokens de padding (-100)
            mask = labels != -100
            predictions = predictions[mask]
            labels = labels[mask]
            
            # Calcular perplexidade
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(torch.tensor(predictions), torch.tensor(labels))
            perplexity = torch.exp(loss).item()
            
            return {"perplexity": perplexity}
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return {"perplexity": float('inf')}
    
    def train_with_curriculum(self, train_data, valid_data=None, use_wandb=False):
        """Treina com curriculum learning"""
        logger.info("Iniciando treinamento com curriculum learning")
        
        # Ordenar dados por dificuldade (comprimento como proxy)
        sorted_data = sorted(train_data, key=len)
        
        # Dividir em estágios
        stages = 3
        stage_size = len(sorted_data) // stages
        
        for stage in range(stages):
            start_idx = stage * stage_size
            end_idx = (stage + 1) * stage_size if stage < stages - 1 else len(sorted_data)
            
            stage_data = sorted_data[start_idx:end_idx]
            logger.info(f"Estágio {stage + 1}/{stages}: treinando com {len(stage_data)} amostras")
            
            result = self.train_supervised(
                stage_data, 
                valid_data, 
                use_wandb=use_wandb,
                num_train_epochs=1
            )
            
            if not result["success"]:
                logger.error(f"Falha no estágio {stage + 1}")
                return result
        
        return {"success": True, "stages_completed": stages}

    def _start_memory_monitoring(self):
        """Inicia monitoramento de memória"""
        try:
            import psutil
            import threading
            
            def monitor():
                while hasattr(self, '_monitoring') and self._monitoring:
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 90:
                        logger.warning(f"Uso de memória alto: {memory_percent}%")
                    time.sleep(10)
            
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=monitor)
            self._monitor_thread.start()
        except ImportError:
            logger.warning("psutil não disponível, monitoramento de memória desabilitado")
    
    def _compute_causal_lm_loss_wrapper(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Wrapper para compatibilidade com diferentes versões do Trainer"""
        return self._compute_causal_lm_loss(model, inputs, return_outputs)
    
    def update_with_feedback(self, use_wandb=False):
        """Atualiza o modelo com dados de feedback"""
        try:
            # Verificar se há feedback suficiente
            if not self.feedback.needs_update():
                logger.info("Feedback insuficiente para atualização")
                return {"success": False, "reason": "insufficient_feedback"}
            
            # Obter dados de alta qualidade
            feedback_data = self.feedback.get_training_data()
            
            if not feedback_data:
                logger.warning("Nenhum dado de feedback de alta qualidade encontrado")
                return {"success": False, "reason": "no_quality_feedback"}
            
            logger.info(f"Atualizando modelo com {len(feedback_data)} exemplos de feedback")
            
            # Treinar com dados de feedback
            result = self.train_supervised(
                feedback_data,
                use_wandb=use_wandb,
                num_train_epochs=1
            )
            
            if result["success"]:
                # Limpar feedback usado
                self.feedback.clear_feedback()
                logger.info("Modelo atualizado com feedback")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro durante atualização com feedback: {e}")
            return {"success": False, "error": str(e)}

    def evaluate_model(self, test_texts):
        """Avalia o modelo com dados de teste"""
        try:
            test_dataset = SupervisedDataset(test_texts, self.tokenizer, max_length=512)
            
            # Criar trainer apenas para avaliação
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            )
            
            trainer = Trainer(
                model=self.model.model if hasattr(self.model, 'model') else self.model,
                args=training_args,
                eval_dataset=test_dataset,
                compute_metrics=self._compute_metrics
            )
            
            eval_results = trainer.evaluate()
            logger.info(f"Resultados da avaliação: {eval_results}")
            return eval_results
            
        except Exception as e:
            logger.error(f"Erro durante avaliação: {e}")
            return {"error": str(e)}

    def _load_model(self):
        """Carrega modelo e tokenizer"""
        try:
            # Carregar modelo
            model = LunaModel.from_pretrained(self.model_dir, self.config.model)
            
            # Carregar tokenizer
            tokenizer = LunaTokenizer(self.config)
            tokenizer_path = os.path.join(self.model_dir, "tokenizer")
            tokenizer.load_from_directory(tokenizer_path)
            
            return model, tokenizer.tokenizer
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
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
        """Treina com otimização automática de hiperparâmetros"""
        try:
            logger.info("Iniciando treinamento com AutoML")
            
            # Configurar AutoML
            automl = LunaAutoML(
                study_name=f"automl_{self.model_name}",
                n_trials=automl_config.get('n_trials', 10) if automl_config else 10,
                metric="eval_loss",
                direction="minimize"
            )
            
            # Função objetivo para otimização
            def objective_function(params):
                # Atualizar configuração com parâmetros otimizados
                self.config.training.learning_rate = params['learning_rate']
                self.config.training.per_device_train_batch_size = params['batch_size']
                self.config.training.weight_decay = params['weight_decay']
                
                # Treinar com parâmetros atuais
                result = self.train_supervised(
                    train_data,
                    valid_data,
                    use_wandb=False,  # Não usar wandb durante otimização
                    num_train_epochs=num_train_epochs or 1,
                    output_dir=os.path.join(base_output_dir or self.output_dir, f"trial_{params.get('trial_id', 0)}")
                )
                
                if not result["success"]:
                    return float('inf')  # Penalidade por falha
                
                # Avaliar modelo
                eval_results = self.evaluate_model(valid_data or train_data[:100])
                return eval_results.get('eval_perplexity', float('inf'))
            
            # Executar otimização
            best_params = automl.optimize(objective_function)
            
            logger.info(f"Melhores parâmetros encontrados: {best_params}")
            
            # Treinar modelo final com melhores parâmetros
            self.config.training.learning_rate = best_params['learning_rate']
            self.config.training.per_device_train_batch_size = best_params['batch_size']
            self.config.training.weight_decay = best_params['weight_decay']
            
            final_result = self.train_supervised(
                train_data,
                valid_data,
                use_wandb=use_wandb,
                num_train_epochs=num_train_epochs
            )
            
            final_result['best_params'] = best_params
            return final_result
            
        except Exception as e:
            logger.error(f"Erro durante treinamento com AutoML: {e}")
            return {"success": False, "error": str(e)}