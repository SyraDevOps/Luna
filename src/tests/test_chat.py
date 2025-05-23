import unittest
import os
from unittest.mock import patch, MagicMock

from src.config.config import Config
from src.chat.luna_chat import LunaChat
from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer

class TestChat(unittest.TestCase):
    """Testes para o módulo de chat"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        self.config = Config()
        self.model_name = "test_model"
        self.model_dir = os.path.join("models", self.model_name)
        
    @patch('src.models.luna_model.LunaModel.from_pretrained')
    @patch('src.models.tokenizer.LunaTokenizer.load_from_directory')
    def test_chat_initialization(self, mock_tokenizer_load, mock_model_from_pretrained):
        """Testa a inicialização do chat"""
        # Configurar mocks
        mock_model = MagicMock()
        mock_model.to_appropriate_device.return_value = "cpu"
        mock_model_from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Mock adicionais para componentes do chat
        with patch('src.models.memory_system.MemorySystem') as mock_memory, \
             patch('src.chat.proactive_messenger.ProactiveMessenger') as mock_proactive, \
             patch('src.models.adaptive_tokenizer.AdaptiveTokenizer') as mock_adaptive, \
             patch('os.path.exists', return_value=True):
            
            # Configurar mocks
            mock_memory_instance = MagicMock()
            mock_memory.return_value = mock_memory_instance
            
            mock_proactive_instance = MagicMock()
            mock_proactive.return_value = mock_proactive_instance
            
            mock_adaptive_instance = MagicMock()
            mock_adaptive.return_value = mock_adaptive_instance
            
            # Criar instância do chat
            chat = LunaChat(self.model_name, self.config, "default")

            # Verificar se foi inicializado corretamente
            self.assertEqual(chat.model_name, "test_model")
            self.assertEqual(chat.persona, "default")
            self.assertFalse(chat.fallback_mode)
            
            # Verificar se o tokenizador foi carregado
            mock_tokenizer_load.assert_called()
    
    @patch('src.models.luna_model.LunaModel.from_pretrained')
    @patch('src.models.tokenizer.LunaTokenizer.load_from_directory')
    def test_extract_response(self, mock_tokenizer_load, mock_model_from_pretrained):
        """Testa o método de extração de resposta"""
        # Configurar mocks
        mock_model = MagicMock()
        mock_model.to_appropriate_device.return_value = "cpu"
        mock_model_from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Mock adicionais
        with patch('src.models.memory_system.MemorySystem'), \
             patch('src.chat.proactive_messenger.ProactiveMessenger'), \
             patch('src.models.adaptive_tokenizer.AdaptiveTokenizer'), \
             patch('os.path.exists', return_value=True):
            
            # Criar instância do chat
            chat = LunaChat(self.model_name, self.config)
            
            # Testar extração de resposta em diferentes cenários
            prompt = "Olá, como vai?"
            
            # Caso 1: Full response contém o prompt no início
            full_response = "Olá, como vai? Estou bem, obrigado!"
            response = chat._extract_response(prompt, full_response)
            self.assertEqual(response, "Estou bem, obrigado!")

    @patch('src.models.luna_model.LunaModel.from_pretrained')
    @patch('src.models.tokenizer.LunaTokenizer.load_from_directory')
    def test_apply_persona_style(self, mock_tokenizer_load, mock_model_from_pretrained):
        """Testa a aplicação de estilos de persona"""
        # Configurar mocks
        mock_model = MagicMock()
        mock_model.to_appropriate_device.return_value = "cpu"
        mock_model_from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Mock adicionais
        with patch('src.models.memory_system.MemorySystem'), \
             patch('src.chat.proactive_messenger.ProactiveMessenger'), \
             patch('src.models.adaptive_tokenizer.AdaptiveTokenizer'), \
             patch('os.path.exists', return_value=True):
            
            # Criar instância do chat
            chat = LunaChat(self.model_name, self.config, persona="casual")
            
            # Testar aplicação de persona
            prompt = "Como você está?"
            styled_prompt = chat._apply_persona_style(prompt)
            
            # Verificar que o prompt foi modificado para incluir persona
            self.assertIsInstance(styled_prompt, str)
            self.assertIn("casual", styled_prompt.lower())

    @patch('src.models.luna_model.LunaModel.from_pretrained')
    @patch('src.models.tokenizer.LunaTokenizer.load_from_directory')
    def test_generate_response_error_handling(self, mock_tokenizer_load, mock_model_from_pretrained):
        """Testa o tratamento de erro na geração de resposta"""
        # Configurar mocks para simular erro
        mock_model = MagicMock()
        mock_model.to_appropriate_device.return_value = "cpu"
        mock_model.generate.side_effect = Exception("Erro simulado")
        mock_model_from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Mock adicionais
        with patch('src.models.memory_system.MemorySystem'), \
             patch('src.chat.proactive_messenger.ProactiveMessenger'), \
             patch('src.models.adaptive_tokenizer.AdaptiveTokenizer'), \
             patch('os.path.exists', return_value=True):
            
            # Criar instância do chat
            chat = LunaChat(self.model_name, self.config)
            
            # Testar tratamento de erro
            response = chat.generate_response("Teste")
            
            # Deve retornar uma resposta de fallback em caso de erro
            self.assertIsInstance(response, str)

if __name__ == "__main__":
    unittest.main()