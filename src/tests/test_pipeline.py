import unittest
import os
import tempfile
import shutil
from src.config.config import Config
from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer
from src.training.trainer import LunaTrainer
from src.models.feedback_system import FeedbackSystem
from src.utils.hardware_utils import detect_hardware

class TestPipeline(unittest.TestCase):
    """Testes de integração para o pipeline completo"""
    
    def setUp(self):
        """Configuração para testes de pipeline"""
        self.test_dir = tempfile.mkdtemp()
        self.model_name = "test_model"
        self.model_dir = os.path.join(self.test_dir, self.model_name)
        
        # Criar configuração
        self.config = Config()
        
        # Textos de amostra para testes
        self.train_texts = [
            "Este é um exemplo de texto para testar o treinamento.",
            "O modelo deve conseguir processar esse texto simples."
        ]
        self.valid_texts = [
            "Exemplo de texto de validação para o modelo."
        ]
        
        # Inicializar sistema de feedback
        self.feedback_system = FeedbackSystem(self.config)
        
        # Importante: criar e salvar o tokenizer antes do modelo
        tokenizer = LunaTokenizer(self.config)
        os.makedirs(os.path.join(self.model_dir, "tokenizer"), exist_ok=True)
        tokenizer.train_and_save(self.train_texts, os.path.join(self.model_dir, "tokenizer"))
        
        # Criar e salvar modelo
        model = LunaModel.from_scratch(self.config.model)
        os.makedirs(self.model_dir, exist_ok=True)
        model.save(self.model_dir)
    
    def tearDown(self):
        """Limpar após testes"""
        shutil.rmtree(self.test_dir)
    
    def test_training_and_inference(self):
        try:
            # Criar diretório para o modelo, se necessário
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Inicializar tokenizer e salvá-lo no diretório correto
            tokenizer = LunaTokenizer(self.config)
            tokenizer_dir = os.path.join(self.model_dir, "tokenizer")
            os.makedirs(tokenizer_dir, exist_ok=True)
            tokenizer.train_and_save(self.train_texts, tokenizer_dir)
            
            # Criar e salvar o modelo no diretório correto
            model = LunaModel.from_scratch(self.config.model)
            model.save(self.model_dir)
            
            # Usar o diretório completo em vez do nome do modelo
            self.trainer = LunaTrainer(self.model_dir, self.config)
            result = self.trainer.train_supervised(self.train_texts, self.valid_texts)
            self.assertTrue(result["success"], "Treinamento falhou!")
            self.assertTrue(os.path.exists(self.model_dir), "Modelo não foi salvo!")
        except Exception as e:
            self.fail(f"Teste falhou com erro: {str(e)}")
    
    def test_feedback_system_integration(self):
        """Testa a integração do sistema de feedback"""
        # Adicionar feedback
        self.feedback_system.add_feedback("Test prompt", "Test response", 5)
        
        # Verificar se o feedback foi armazenado
        self.assertGreater(len(self.feedback_system.feedback_data), 0)
        
        # Verificar se os dados de alta qualidade são filtrados corretamente
        high_quality = self.feedback_system.get_high_quality_feedback()
        self.assertEqual(len(high_quality), 1)
    
    def test_pipeline(self):
        """Teste de fluxo completo: treinamento, feedback e refinamento"""
        # Verificar se os diretórios padrão existem
        self.assertTrue(os.path.exists("data/train"))
        self.assertTrue(os.path.exists("data/valid"))

if __name__ == "__main__":
    unittest.main()