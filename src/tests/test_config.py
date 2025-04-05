import unittest
import os
import tempfile
import json
from src.config.config import Config

class TestConfig(unittest.TestCase):
    """Testes unitários para o sistema de configuração"""
    
    def setUp(self):
        """Configuração para cada teste"""
        # Create a temporary config file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, 'test_config.json')
        
        # Sample configuration data
        self.config_data = {
            "model": {
                "vocab_size": 5000,
                "hidden_size": 128,
                "num_hidden_layers": 2
            },
            "training": {
                "learning_rate": 5e-5,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 2
            },
            "persona": {
                "default": "casual"
            }
        }
        
        # Write sample config to file
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f)
    
    def tearDown(self):
        """Limpeza após cada teste"""
        self.temp_dir.cleanup()
    
    def test_config_loading(self):
        """Testar carregamento básico de configuração"""
        config = Config(config_path=self.config_path)
        self.assertEqual(config.model.vocab_size, 5000)
        self.assertEqual(config.training.learning_rate, 5e-5)
        self.assertEqual(config.persona.default, "casual")
    
    def test_config_defaults(self):
        """Testar valores padrão quando não especificados"""
        # Create a minimal config file
        minimal_config = {"model": {"vocab_size": 1000}}
        minimal_path = os.path.join(self.temp_dir.name, 'minimal_config.json')
        with open(minimal_path, 'w') as f:
            json.dump(minimal_config, f)
        
        # Load and test
        config = Config(config_path=minimal_path)
        self.assertEqual(config.model.vocab_size, 1000)
        # Test that other values get defaults
        self.assertTrue(hasattr(config.training, 'learning_rate'))
    
    def test_config_validation(self):
        """Testar validação de configuração"""
        # Criar uma configuração com valores inválidos para testar validação
        config = Config()
        
        # Forçar um valor inválido para testar validação
        config.model.vocab_size = -100
        
        # Chamar validação explicitamente
        config = config.validate()
        
        # Verificar se a validação corrigiu o valor
        self.assertGreater(config.model.vocab_size, 0)

if __name__ == '__main__':
    unittest.main()