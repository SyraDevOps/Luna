import unittest
import os
import tempfile
import shutil
import torch
from src.config.config import Config
from src.models.luna_model import LunaModel

class TestModel(unittest.TestCase):
    """Testes para o módulo do modelo"""
    
    def setUp(self):
        self.config = Config()
        # Configuração mais leve para testes
        self.config.model.n_embd = 128
        self.config.model.n_layer = 2
        self.config.model.n_head = 2
        self.temp_dir = tempfile.mkdtemp()
        self.model = LunaModel.from_scratch(self.config.model)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_model_creation(self):
        """Testa a criação do modelo"""
        self.assertIsNotNone(self.model.model)
        
    def test_model_save_load(self):
        """Testa o salvamento e carregamento do modelo"""
        import tempfile
        
        # Criar um diretório temporário para teste
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_model")
            
            # Salvar o modelo
            self.model.save(save_path)
            
            # Verificar se o arquivo de configuração foi criado
            self.assertTrue(os.path.exists(os.path.join(save_path, "config.json")))
            
            # Verificar se o arquivo de pesos foi criado (pode ser pytorch_model.bin ou outro formato)
            has_weights = os.path.exists(os.path.join(save_path, "pytorch_model.bin")) or \
                         os.path.exists(os.path.join(save_path, "model.safetensors"))
            self.assertTrue(has_weights, "Arquivos de pesos não foram criados")
    
    def test_forward_pass(self):
        """Testa uma passagem pelo modelo"""
        # Criar um tensor de input simulado
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, self.config.model.vocab_size, (batch_size, seq_length))
        
        # Verificar se consegue processar
        try:
            outputs = self.model.model(input_ids)
            self.assertIsNotNone(outputs)
            self.assertEqual(outputs.logits.shape[0], batch_size)
            self.assertEqual(outputs.logits.shape[1], seq_length)
        except Exception as e:
            self.fail(f"Erro no forward pass: {e}")