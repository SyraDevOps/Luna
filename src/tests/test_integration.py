import unittest
import os
import tempfile
import shutil
from src.config.config import Config
from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer
from src.training.trainer import LunaTrainer
from src.models.feedback_system import FeedbackSystem

class TestIntegration(unittest.TestCase):
    """Testes de integração para o sistema completo"""
    
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_dir = os.path.join(cls.temp_dir, "integration_test_model")
        cls.config = Config()
        cls.config.model.n_layer = 2  # Reduzir tamanho para testes
        cls.config.model.n_head = 2
        cls.config.model.n_embd = 32
        cls.config.model.tokenizer_path = os.path.join(cls.temp_dir, "tokenizer")
        
        # Dados de exemplo
        cls.train_data = [
            "Este é um exemplo de treino.",
            "LunaGPT é um sistema de diálogo.",
            "Pergunta: Como testar modelos?\nResposta: Usando testes de integração."
        ]
        
        cls.valid_data = [
            "Exemplo de validação.",
            "Pergunta: O que é teste?\nResposta: Verificação de funcionamento."
        ]
        
        # Criar diretórios e arquivos para teste
        os.makedirs(cls.model_dir, exist_ok=True)
        
        # Configurar para teste
        cls.config.tokenizer_training_data = cls.train_data
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_full_pipeline(self):
        """Testa o pipeline completo: criar, treinar, feedback e refinar"""
        # 1. Criar tokenizer
        tokenizer = LunaTokenizer(self.config)
        self.assertIsNotNone(tokenizer.tokenizer)
        
        # 2. Criar modelo
        model = LunaModel(self.config.model)
        self.assertIsNotNone(model)
        
        # 3. Salvar modelo
        model.save(self.model_dir)
        tokenizer_path = os.path.join(self.model_dir, "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.tokenizer.save_pretrained(tokenizer_path)
        
        # 4. Verificar se arquivos foram criados
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")))
        
        # 5. Teste de feedback
        feedback_config = self.config
        feedback_config.feedback.feedback_file = os.path.join(self.temp_dir, "feedback.jsonl")
        
        feedback_system = FeedbackSystem(feedback_config)
        feedback_system.add_feedback("Teste integrado?", "Sim, funcionando.", 5)
        self.assertEqual(len(feedback_system.feedback), 1)
        self.assertTrue(os.path.exists(feedback_config.feedback.feedback_file))
        
        # 6. Verificar necessidade de atualização
        feedback_system.config.min_samples_for_update = 1
        self.assertTrue(feedback_system.needs_update())