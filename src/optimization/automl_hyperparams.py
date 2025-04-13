import os
import logging
import numpy as np
import optuna
from optuna.trial import Trial
from functools import partial
import json
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional, Union

logger = logging.getLogger(__name__)

class LunaAutoML:
    """
    Sistema de otimização automática de hiperparâmetros para LunaGPT
    usando Optuna para busca bayesiana eficiente.
    """
    
    def __init__(
        self, 
        study_name: str = "lunagpt_automl",
        storage_path: str = None,
        metric: str = "validation_loss",
        direction: str = "minimize",
        n_trials: int = 20,
        n_jobs: int = 1,
        pruner_type: str = "hyperband",
    ):
        """
        Inicializa o sistema de AutoML.
        
        Args:
            study_name: Nome do estudo para rastreamento
            storage_path: Caminho para armazenar os resultados (sqlite ou None para memória)
            metric: Métrica a ser otimizada
            direction: "minimize" ou "maximize"
            n_trials: Número máximo de tentativas
            n_jobs: Paralelismo (-1 para usar todos os cores)
            pruner_type: Tipo de pruner ("hyperband", "median", "threshold", "none")
        """
        self.study_name = study_name
        self.storage_path = storage_path
        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        
        # Configurar armazenamento
        if storage_path:
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            self.storage = f"sqlite:///{storage_path}"
        else:
            self.storage = None
            
        # Configurar pruner (elimina tentativas ruins previamente)
        if pruner_type == "hyperband":
            self.pruner = optuna.pruners.HyperbandPruner()
        elif pruner_type == "median":
            self.pruner = optuna.pruners.MedianPruner()
        elif pruner_type == "threshold":
            self.pruner = optuna.pruners.ThresholdPruner(upper=0.1 if direction == "minimize" else 0.9)
        else:
            self.pruner = optuna.pruners.NopPruner()
            
        # Inicializar estudo
        self.study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction=direction,
            pruner=self.pruner,
            load_if_exists=True
        )
        
        # Espaço de parâmetros padrão
        self.param_space = {}
        self.default_params = {}
        self._define_default_param_space()
        
    def _define_default_param_space(self):
        """Define o espaço de busca padrão para hiperparâmetros comuns"""
        self.add_param_space(
            "learning_rate", 
            param_type="float", 
            low=1e-6, high=1e-3, 
            log=True,
            default=5e-5
        )
        self.add_param_space(
            "weight_decay", 
            param_type="float", 
            low=0.0, high=0.1, 
            default=0.01
        )
        self.add_param_space(
            "warmup_ratio", 
            param_type="float", 
            low=0.0, high=0.2, 
            default=0.1
        )
        self.add_param_space(
            "dropout_rate", 
            param_type="float", 
            low=0.0, high=0.3, 
            default=0.1
        )
        self.add_param_space(
            "attention_dropout", 
            param_type="float", 
            low=0.0, high=0.2, 
            default=0.1
        )
        self.add_param_space(
            "batch_size", 
            param_type="categorical", 
            choices=[1, 2, 4, 8, 16], 
            default=4
        )
        self.add_param_space(
            "gradient_accumulation_steps", 
            param_type="categorical", 
            choices=[1, 2, 4, 8, 16], 
            default=1
        )
        
    def add_param_space(
        self, 
        param_name: str, 
        param_type: str, 
        low: float = None, 
        high: float = None, 
        choices: List = None, 
        log: bool = False,
        default: Any = None
    ):
        """
        Adiciona um parâmetro ao espaço de busca
        
        Args:
            param_name: Nome do parâmetro
            param_type: Tipo ("float", "int", "categorical")
            low: Valor mínimo (para float/int)
            high: Valor máximo (para float/int)
            choices: Lista de opções (para categorical)
            log: Se True, usa escala logarítmica para busca (para float)
            default: Valor padrão do parâmetro
        """
        self.param_space[param_name] = {
            "type": param_type,
            "low": low,
            "high": high,
            "choices": choices,
            "log": log
        }
        
        if default is not None:
            self.default_params[param_name] = default
            
    def sample_params(self, trial: Trial) -> Dict[str, Any]:
        """Amostra parâmetros do espaço de busca para uma tentativa específica"""
        params = {}
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config["type"]
            
            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config["low"], 
                    param_config["high"], 
                    log=param_config["log"]
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"]
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
                
        return params
    
    def objective(self, trial: Trial, train_fn: Callable, **train_kwargs) -> float:
        """
        Função objetivo para otimização
        
        Args:
            trial: Tentativa atual do Optuna
            train_fn: Função de treinamento que aceita hiperparâmetros
            train_kwargs: Argumentos adicionais para a função de treinamento
            
        Returns:
            float: Valor da métrica a ser otimizada
        """
        # Obter hiperparâmetros para esta tentativa
        params = self.sample_params(trial)
        
        # Mesclar com argumentos existentes
        full_kwargs = {**train_kwargs, **params}
        
        # Executar treinamento
        result = train_fn(**full_kwargs)
        
        # Extrair e retornar a métrica
        if isinstance(result, dict):
            if self.metric in result:
                return result[self.metric]
            else:
                raise ValueError(f"Métrica '{self.metric}' não encontrada nos resultados.")
        else:
            return result
            
    def optimize(self, train_fn: Callable, **train_kwargs) -> Dict[str, Any]:
        """
        Executa a otimização de hiperparâmetros
        
        Args:
            train_fn: Função de treinamento que aceita hiperparâmetros
            train_kwargs: Argumentos adicionais para a função de treinamento
            
        Returns:
            Dict: Melhores hiperparâmetros encontrados
        """
        objective_with_kwargs = partial(self.objective, train_fn=train_fn, **train_kwargs)
        
        self.study.optimize(objective_with_kwargs, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info(f"Melhor valor ({self.metric}): {best_value}")
        logger.info(f"Melhores hiperparâmetros: {best_params}")
        
        # Salvar resultados
        self._save_results()
        
        return best_params
    
    def _save_results(self):
        """Salva os resultados da otimização"""
        results_dir = os.path.dirname(self.storage_path) if self.storage_path else "automl_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(results_dir, f"{self.study_name}_{timestamp}.json")
        
        trials = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials.append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "datetime": trial.datetime.isoformat() if trial.datetime else None,
                    "duration": trial.duration.total_seconds() if trial.duration else None,
                })
        
        results = {
            "study_name": self.study_name,
            "direction": self.direction,
            "metric": self.metric,
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "trials": trials,
        }
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Resultados salvos em: {results_path}")

    def get_default_params(self):
        """Retorna os parâmetros padrão"""
        return self.default_params.copy()
        
    def get_best_params(self):
        """Retorna os melhores parâmetros encontrados"""
        if hasattr(self.study, "best_params"):
            return self.study.best_params
        else:
            return self.default_params.copy()


class DynamicHyperparamOptimizer:
    """
    Otimizador dinâmico que ajusta hiperparâmetros durante o treinamento
    com base em métricas observadas.
    """
    
    def __init__(
        self, 
        initial_params: Dict[str, Any],
        update_interval: int = 100, 
        patience: int = 3,
        max_adjustments: int = 5
    ):
        """
        Inicializa o otimizador dinâmico
        
        Args:
            initial_params: Hiperparâmetros iniciais
            update_interval: Intervalo de steps para verificar ajustes
            patience: Número de intervalos sem melhoria antes de ajustar
            max_adjustments: Número máximo de ajustes por treinamento
        """
        self.params = initial_params.copy()
        self.update_interval = update_interval
        self.patience = patience
        self.max_adjustments = max_adjustments
        
        self.history = []
        self.no_improvement_count = 0
        self.adjustment_count = 0
        self.best_metric = float("inf")  # Para métricas que minimizamos (como loss)
        
    def update(self, metrics: Dict[str, float], global_step: int) -> Dict[str, Any]:
        """
        Atualiza hiperparâmetros com base nas métricas atuais
        
        Args:
            metrics: Métricas observadas (dict)
            global_step: Step atual do treinamento
            
        Returns:
            Dict: Hiperparâmetros atualizados ou None se não houver mudanças
        """
        # Verificar se é momento de ajustar
        if global_step % self.update_interval != 0:
            return None
            
        # Registrar métricas
        self.history.append(metrics)
        
        # Extrair métrica principal (assumindo loss)
        current_metric = metrics.get("loss", metrics.get("validation_loss", None))
        if current_metric is None:
            return None
            
        # Verificar se houve melhoria
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            
        # Se não houver melhoria por vários intervalos e ainda temos ajustes disponíveis
        if self.no_improvement_count >= self.patience and self.adjustment_count < self.max_adjustments:
            # Ajustar learning rate (exemplo de ajuste)
            if "learning_rate" in self.params:
                self.params["learning_rate"] *= 0.5  # Reduzir learning rate pela metade
                
            # Aumentar batch size se possível (exemplo de ajuste)
            if "batch_size" in self.params and self.params["batch_size"] < 32:
                next_batch_size = self.params["batch_size"] * 2
                self.params["batch_size"] = min(next_batch_size, 32)
                
            # Registrar ajuste
            self.adjustment_count += 1
            self.no_improvement_count = 0
            
            logger.info(f"Ajuste dinâmico de hiperparâmetros no step {global_step}: {self.params}")
            return self.params
            
        return None