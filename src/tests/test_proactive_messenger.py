import unittest
from unittest.mock import MagicMock, patch
import time

from src.chat.proactive_messenger import ProactiveMessenger

class TestProactiveMessenger(unittest.TestCase):
    """Testes para o sistema de mensagens proativas"""
    
    def setUp(self):
        """Configuração para os testes"""
        self.chat_instance = MagicMock()
        self.chat_instance.conversation_history = []
        self.messenger = ProactiveMessenger(self.chat_instance, inactivity_threshold=0.1)
        
    def tearDown(self):
        """Limpeza após os testes"""
        if hasattr(self.messenger, 'monitoring_thread') and self.messenger.monitoring_thread:
            self.messenger.stop_monitoring()
    
    def test_register_callback(self):
        """Testa o registro de callback"""
        callback = MagicMock()
        self.messenger.register_suggestion_callback(callback)
        self.assertEqual(self.messenger.suggestion_callback, callback)
    
    def test_activity_timer(self):
        """Testa o reset do timer de atividade"""
        old_time = self.messenger.last_activity_time
        time.sleep(0.01)  # Pequena pausa para garantir que o tempo mudou
        self.messenger.reset_activity_timer()
        new_time = self.messenger.last_activity_time
        self.assertGreater(new_time, old_time)
    
    def test_proactive_suggestion_generation(self):
        """Testa a geração de sugestões proativas baseadas em padrões"""
        # Testar padrão de indecisão
        self.chat_instance.conversation_history = [{"user": "Não sei o que fazer"}]
        suggestion = self.messenger._get_proactive_suggestion()
        self.assertIsNotNone(suggestion)
        
        # Testar padrão de ajuda
        self.chat_instance.conversation_history = [{"user": "Preciso de ajuda com isso"}]
        suggestion = self.messenger._get_proactive_suggestion()
        self.assertIsNotNone(suggestion)
        
        # Testar padrão de pesquisa
        self.chat_instance.conversation_history = [{"user": "Onde posso encontrar informações?"}]
        suggestion = self.messenger._get_proactive_suggestion()
        self.assertIsNotNone(suggestion)
        
        # Testar texto sem padrão específico
        self.chat_instance.conversation_history = [{"user": "Vamos falar de outro assunto"}]
        suggestion = self.messenger._get_proactive_suggestion()
        self.assertEqual(suggestion, "Posso ajudar com mais alguma coisa?")
    
    @patch('time.sleep', return_value=None)  # Evitar esperas em testes
    def test_monitoring_lifecycle(self, mock_sleep):
        """Testa o ciclo de vida do monitoramento"""
        # Iniciar monitoramento
        self.messenger.start_monitoring()
        self.assertTrue(self.messenger.monitoring_active)
        self.assertIsNotNone(self.messenger.monitoring_thread)
        
        # Parar monitoramento
        self.messenger.stop_monitoring()
        self.assertFalse(self.messenger.monitoring_active)

if __name__ == "__main__":
    unittest.main()