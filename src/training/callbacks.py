from typing import List, Dict
from datetime import datetime
import logging
import time
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

class CustomCheckpointCallback:
    def on_save(self, args, state, control, **kwargs):
        logger.info(f"Checkpoint saved at step {state.global_step}.")

class TrainingTimeCallback:
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        logger.info("Training started.")

    def on_train_end(self, args, state, control, **kwargs):
        duration = datetime.now() - self.start_time
        logger.info(f"Training finished in {duration}.")

class DynamicHyperparamCallback:
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0 and hasattr(state, 'loss'):
            new_lr = args.learning_rate * 0.95
            args.learning_rate = new_lr
            logger.info(f"Dynamic adjustment: new learning rate = {new_lr}")

class MemoryReplayCallback:
    def __init__(self, memory_bank: List[str]):
        self.memory_bank = memory_bank

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.memory_bank:
            random.shuffle(self.memory_bank)
            new_dataset = SupervisedDataset(kwargs["tokenizer"], self.memory_bank)
            kwargs["trainer"].train_dataset = new_dataset
            del new_dataset

class ProgressBarCallback:
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_bar = tqdm(total=args.num_train_epochs, desc="Epochs", leave=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.epoch_bar.close()

class EarlyStoppingCallback:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_metric = float('inf')
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_metric = metrics.get("eval_loss", None)
        if current_metric is None:
            return
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                control.should_training_stop = True
                logger.info("Early stopping triggered!")

class CheckpointPTCallback:
    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        model_dir = trainer.args.output_dir
        try:
            trainer.model.save_pretrained(model_dir)
            pt_path = os.path.join(model_dir, "model.pt")
            torch.save(trainer.model.state_dict(), pt_path)
            alt_path = os.path.join(model_dir, f"model_{os.path.basename(os.path.normpath(model_dir))}.pt")
            torch.save(trainer.model.state_dict(), alt_path)
            logger.info(f"Checkpoint .pt and model_{os.path.basename(os.path.normpath(model_dir))}.pt saved in {model_dir} at epoch {state.epoch}")
        except Exception as e:
            logger.error("Error saving checkpoint .pt: " + str(e))

class CustomTrainingCallback(TrainerCallback):
    def __init__(self, timeout_min=30):
        self.timeout_min = timeout_min
        self.start_time = None
        self.last_step = -1
        self.no_progress_count = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info(f"Iniciando treinamento com timeout de {self.timeout_min} minutos")
    
    def on_step_end(self, args, state, control, **kwargs):
        # Verificar timeout global
        elapsed_minutes = (time.time() - self.start_time) / 60.0
        if elapsed_minutes > self.timeout_min:
            logger.warning(f"Treinamento interrompido por timeout após {elapsed_minutes:.2f} minutos")
            control.should_training_stop = True
            return
        
        # Verificar progresso entre steps
        if state.global_step == self.last_step:
            self.no_progress_count += 1
            if self.no_progress_count > 3:
                logger.warning(f"Nenhum progresso detectado. Interrompendo treinamento.")
                control.should_training_stop = True
        else:
            self.no_progress_count = 0
            self.last_step = state.global_step
            
            # Registrar métricas do passo atual se disponíveis
            if hasattr(state, "log_history") and state.log_history:
                latest_metrics = state.log_history[-1]
                logger.info(f"Passo {state.global_step}: loss={latest_metrics.get('loss', 'N/A')}")