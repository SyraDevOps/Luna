import unittest
import os
import tempfile
import shutil
import torch  # Add the missing import
from src.training.trainer import LunaTrainer
from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer
from src.config.config import Config

class TestTraining(unittest.TestCase):
    """Testes para o sistema de treinamento"""
    
    def setUp(self):
        """Configuração para os testes"""
        self.config = Config()
        self.test_model_name = "test_model"
        self.test_model_dir = os.path.join("models", self.test_model_name)
        
        # Criar modelo para teste
        if not os.path.exists(self.test_model_dir):
            os.makedirs(os.path.join(self.test_model_dir, "tokenizer"), exist_ok=True)
            
            # Criar modelo e tokenizer mínimos
            texts = ["Este é um texto de exemplo para teste."]
            tokenizer = LunaTokenizer(self.config)
            tokenizer.train_and_save(texts, os.path.join(self.test_model_dir, "tokenizer"))
            
            model = LunaModel.from_scratch(self.config.model)
            model.save(self.test_model_dir)
        
        self.trainer = LunaTrainer(self.test_model_name, self.config)
        
    def tearDown(self):
        """Limpeza após os testes"""
        pass
        
    def test_train_supervised_minimal(self):
        """Testa treinamento supervisionado mínimo"""
        train_texts = [
            "Pergunta: O que é LunaGPT?\nResposta: LunaGPT é um sistema de diálogo adaptativo em português."
        ]
        
        result = self.trainer.train_supervised(train_texts)
        
        # Verificar se o treinamento teve sucesso
        self.assertTrue(result['success'])
        
    def test_compute_loss(self):
        """Testa o cálculo de perda personalizado"""
        # Criar entradas de teste
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }
        
        # Testar cálculo de perda
        loss = self.trainer._compute_causal_lm_loss_wrapper(self.trainer.model.model, inputs)
        
        # Verificar se a perda é um valor de tensor
        self.assertTrue(isinstance(loss, torch.Tensor))

class TestDynamicAdjustments(unittest.TestCase):
    def test_adjust_model_config_dynamically(self):
        hardware_profile = Mock(system_type="low-end")
        config = Mock(num_hidden_layers=12, num_attention_heads=12, hidden_size=768, n_positions=1024)
        adjusted_config = adjust_model_config_dynamically(config, hardware_profile)
        self.assertEqual(adjusted_config.num_hidden_layers, 4)
        self.assertEqual(adjusted_config.hidden_size, 256)

def test_training_and_saving():
    trainer = LunaTrainer("test_model", config)
    result = trainer.train_supervised(["Exemplo de texto de treinamento."])
    assert result["success"], "Treinamento falhou!"
    assert os.path.exists("models/test_model/pytorch_model.bin"), "Pesos não foram salvos!"