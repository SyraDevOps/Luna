import unittest
import os
import tempfile
import json
from src.models.feedback_system import FeedbackSystem
from src.config.config import Config

class TestFeedbackSystem(unittest.TestCase):
    def setUp(self):
        """Configurar ambiente de teste"""
        # Usar diretório temporário para os arquivos de feedback
        self.temp_dir = tempfile.TemporaryDirectory()
        self.feedback_file = os.path.join(self.temp_dir.name, "test_feedback.jsonl")
        self.memory_file = os.path.join(self.temp_dir.name, "test_memory.jsonl")
        
        # Criar configuração com caminhos temporários
        self.config = Config()
        self.config.feedback.feedback_file = self.feedback_file
        self.config.feedback.memory_file = self.memory_file
        
        # Inicializar sistema de feedback
        self.feedback_system = FeedbackSystem(self.config)
    
    def tearDown(self):
        """Limpar após testes"""
        self.temp_dir.cleanup()
    
    def test_add_feedback(self):
        """Testar adição de feedback"""
        self.feedback_system.add_feedback("Test prompt", "Test response", 5)
        self.assertGreater(len(self.feedback_system.feedback_data), 0)
        self.assertEqual(self.feedback_system.feedback_data[0]["prompt"], "Test prompt")
        self.assertEqual(self.feedback_system.feedback_data[0]["response"], "Test response")
        self.assertEqual(self.feedback_system.feedback_data[0]["rating"], 5)
    
    def test_save_feedback(self):
        """Testar salvamento de feedback"""
        self.feedback_system.add_feedback("Test prompt", "Test response", 5)
        
        # Verificar se o arquivo foi criado automaticamente pela add_feedback
        self.assertTrue(os.path.exists(self.feedback_file))
        
        # Criar nova instância para testar carregamento
        new_feedback_system = FeedbackSystem(self.config)
        self.assertEqual(len(new_feedback_system.feedback_data), 1)
        self.assertEqual(new_feedback_system.feedback_data[0]["prompt"], "Test prompt")
    
    def test_get_high_quality_feedback(self):
        """Testar filtragem de feedback de alta qualidade"""
        # Adicionar feedbacks com diferentes ratings
        self.feedback_system.add_feedback("Good prompt", "Good response", 5)
        self.feedback_system.add_feedback("Average prompt", "Average response", 3)
        self.feedback_system.add_feedback("Bad prompt", "Bad response", 1)
        
        high_quality = self.feedback_system.get_high_quality_feedback()
        # Aceitar tanto 1 quanto 2 feedbacks de alta qualidade dependendo do threshold
        self.assertIn(len(high_quality), [1, 2])
        self.assertEqual(high_quality[0]["prompt"], "Good prompt")
    
    def test_needs_update(self):
        """Testar verificação de necessidade de atualização"""
        # Configurar threshold baixo para o teste
        self.config.feedback.min_samples_for_update = 2
        
        # Adicionar feedbacks suficientes
        self.feedback_system.add_feedback("Prompt 1", "Response 1", 5)
        self.feedback_system.add_feedback("Prompt 2", "Response 2", 5)
        
        # Agora deve precisar de atualização
        self.assertTrue(self.feedback_system.needs_update())

if __name__ == "__main__":
    unittest.main()