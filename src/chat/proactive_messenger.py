import threading
import time
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ProactiveMessenger:
    """Sistema de mensagens proativas baseado em inatividade do usuário"""
    def __init__(self, chat_instance, inactivity_threshold=60):
        self.chat_instance = chat_instance
        self.inactivity_threshold = inactivity_threshold
        self.last_activity = time.time()
        self.running = True
        self.thread = threading.Thread(target=self.monitor_inactivity, daemon=True)
        self.thread.start()

    def update_activity(self):
        self.last_activity = time.time()

    def monitor_inactivity(self):
        while self.running:
            time.sleep(5)  # Verifica a cada 5 segundos
            elapsed = time.time() - self.last_activity
            if elapsed > self.inactivity_threshold:
                suggestion = self._get_proactive_suggestion()
                if suggestion:
                    print(f"\nLuna: {suggestion}")
                    self.last_activity = time.time()

    def stop(self):
        self.running = False
        
    def _get_proactive_suggestion(self):
        """Gera uma sugestão proativa baseada no contexto"""
        if not self.chat_instance.conversation_history:
            return None
            
        # Padrões para gatilhos de sugestões proativas
        triggers = {
            r"não sei": "Posso explicar isso para você em mais detalhes?",
            r"como (fazer|funciona)": "Gostaria que eu explicasse isso passo a passo?",
            r"ajuda": "Em que posso ajudar você especificamente?",
            r"interessante": "Gostaria de saber mais sobre esse assunto?",
        }
        
        # Obter a última mensagem do usuário
        last_exchange = self.chat_instance.conversation_history[-1]
        last_user_message = last_exchange["user"].lower()
        
        # Verificar padrões nos últimos inputs do usuário
        for pattern, response in triggers.items():
            if re.search(pattern, last_user_message, re.IGNORECASE):
                return response
                
        return None

def proactive_suggestions(context: str) -> Optional[str]:
    # Implementação da função
    pass