import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from src.config.config import Config
from src.models.luna_model import LunaModel
from src.chat.luna_chat import LunaChat

class TestChat(unittest.TestCase):
    """Testes para o módulo de chat"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        self.config = Config()
        self.model_name = "test_model"
        self.model_dir = os.path.join("models", self.model_name)
        
        # Criar modelo se não existir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            model = LunaModel(self.config.model)
            model.save(self.model_dir)
    
    @patch('src.chat.luna_chat.PreTrainedTokenizerFast')
    @patch('src.chat.luna_chat.GPT2LMHeadModel')
    def test_chat_interaction(self, model_mock, tokenizer_mock):
        """Testa a interação do chat"""
        tokenizer_mock.from_pretrained.return_value = None
        model_mock.from_pretrained.return_value = None

        chat = LunaChat(self.model_name, self.config)
        response = chat.generate_response("Olá, como vai?")
        self.assertIsNotNone(response)