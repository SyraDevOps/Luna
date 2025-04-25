class Config:
    def __init__(self, feedback_file: str = "feedback.jsonl", min_samples_for_update: int = 5):
        self.feedback = FeedbackConfig(feedback_file, min_samples_for_update)
        
        # Configurações para tokenizer adaptativo
        self.enable_tokens_learning = True
        self.min_token_frequency = 5  # Mínimo de ocorrências para adicionar um token
        self.max_new_tokens_per_session = 1000000  # Virtualmente sem limite
        self.tokens_update_frequency = 7  # Atualizar a cada 7 mensagens
        self.tokens_learning_during_training = True  # Ativar durante treinamento
        self.collect_training_tokens = True  # Coletar tokens durante treinamento
        self.token_collection_file = "token_candidates.json"  # Arquivo para armazenar candidatos

class FeedbackConfig:
    def __init__(self, feedback_file: str, min_samples_for_update: int):
        self.feedback_file = feedback_file
        self.min_samples_for_update = min_samples_for_update