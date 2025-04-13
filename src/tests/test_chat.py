import os
import unittest
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
    @patch('src.models.tokenizer.LunaTokenizer.load')
    def test_chat_initialization(self, mock_tokenizer_load, mock_model_from_pretrained):
        """Testa a inicialização do chat"""
        # Configurar mocks
        mock_model = MagicMock()
        mock_model.to_appropriate_device.return_value = "cpu"
        mock_model_from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Criar instância com os mocks
        with patch('os.path.exists', return_value=True):
            chat = LunaChat(self.model_name, self.config)
            
        # Verificações
        self.assertEqual(chat.model_name, self.model_name)
        self.assertEqual(chat.persona, "casual")
        self.assertFalse(chat.fallback_mode)
        mock_model_from_pretrained.assert_called_once()
        mock_tokenizer_load.assert_called_once()
    
    @patch('src.models.luna_model.LunaModel.from_pretrained')
    @patch('src.models.tokenizer.LunaTokenizer.load')
    def test_extract_response(self, mock_tokenizer_load, mock_model_from_pretrained):
        """Testa o método de extração de resposta"""
        # Configurar mocks
        mock_model = MagicMock()
        mock_model.to_appropriate_device.return_value = "cpu"
        mock_model_from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Criar instância do chat
        with patch('os.path.exists', return_value=True):
            chat = LunaChat(self.model_name, self.config)
        
        # Testar extração de resposta em diferentes cenários
        prompt = "Olá, como vai?"
        
        # Caso 1: Full response contém o prompt no início
        full_response = "Olá, como vai? Estou bem, obrigado!"
        response = chat._extract_response(prompt, full_response)
        self.assertEqual(response, "Estou bem, obrigado!")
        
        # Caso 2: Full response contém marcador "Luna:"
        full_response = "Usuário: Como vai?\nLuna: Estou bem, obrigado!"
        response = chat._extract_response(prompt, full_response)
        self.assertEqual(response, "Estou bem, obrigado!")
        
        # Caso 3: Nem prompt nem marcador estão presentes
        full_response = "Estou bem, obrigado por perguntar!"
        response = chat._extract_response(prompt, full_response)
        self.assertEqual(response, "Estou bem, obrigado por perguntar!")
    
    @patch('src.models.luna_model.LunaModel.from_pretrained')
    @patch('src.models.tokenizer.LunaTokenizer.load')
    def test_apply_persona_style(self, mock_tokenizer_load, mock_model_from_pretrained):
        """Testa a aplicação de estilos de persona"""
        # Configurar mocks
        mock_model = MagicMock()
        mock_model.to_appropriate_device.return_value = "cpu"
        mock_model_from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_load.return_value = mock_tokenizer
        
        prompt = "Olá, como vai?"
        
        # Testar diferentes personas
        with patch('os.path.exists', return_value=True):
            # Persona técnica
            chat_tecnico = LunaChat(self.model_name, self.config, persona="tecnico")
            resultado_tecnico = chat_tecnico._apply_persona_style(prompt)
            self.assertIn("[PERSONA: técnico", resultado_tecnico)
            
            # Persona casual (padrão)
            chat_casual = LunaChat(self.model_name, self.config, persona="casual")
            resultado_casual = chat_casual._apply_persona_style(prompt)
            self.assertIn("[PERSONA: amigável", resultado_casual)
            
            # Persona formal
            chat_formal = LunaChat(self.model_name, self.config, persona="formal")
            resultado_formal = chat_formal._apply_persona_style(prompt)
            self.assertIn("[PERSONA: formal", resultado_formal)

    @patch('src.models.luna_model.LunaModel.from_pretrained')
    @patch('src.models.tokenizer.LunaTokenizer.load')
    def test_generate_response_error_handling(self, mock_tokenizer_load, mock_model_from_pretrained):
        """Testa o tratamento de erro na geração de resposta"""
        # Configurar mocks
        mock_model = MagicMock()
        mock_model.to_appropriate_device.return_value = "cpu"
        mock_model_from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_load.return_value = mock_tokenizer
        
        # Criar instância do chat
        with patch('os.path.exists', return_value=True):
            chat = LunaChat(self.model_name, self.config)
        
        # Testar erro durante geração com exceção real (não mock)
        with patch.object(chat, 'model', None):  # Force failure
            response = chat.generate_response("Olá")
            self.assertIn("Desculpe, ocorreu um erro", response)

if __name__ == "__main__":
    unittest.main()