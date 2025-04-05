import os
import torch
import re
import logging
import time
import threading
import traceback
from typing import List, Dict, Optional, Any, Union

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import PreTrainedTokenizerFast  # Add this import
from src.utils.logging_utils import logger
from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer

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
            try:
                elapsed = time.time() - self.last_activity
                if elapsed > self.inactivity_threshold:
                    suggestion = self._get_proactive_suggestion()
                    if suggestion:
                        print(f"\nLuna: {suggestion}")
                        self.last_activity = time.time()
            except Exception as e:
                # Prevenir que exceções derrubem a thread
                logger.error(f"Erro no monitoramento proativo: {e}")

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
            r"\?$": "Tem mais alguma pergunta sobre este tema?",
            r"obrigad[oa]": "Fico feliz em ajudar. O que mais posso fazer por você?",
        }
        
        # Obter a última mensagem do usuário
        last_exchange = self.chat_instance.conversation_history[-1]
        last_user_message = last_exchange["user"].lower()
        
        # Verificar padrões nos últimos inputs do usuário
        for pattern, response in triggers.items():
            if re.search(pattern, last_user_message, re.IGNORECASE):
                return response
                
        # Se não encontrou um gatilho específico, mas passou tempo suficiente
        return "Como posso continuar te ajudando hoje?"

class LunaChat:
    """Interface de chat para interação com o modelo Luna"""
    
    def __init__(self, model_name: str, config, persona: str = "casual"):
        """Inicializa o chat com um modelo treinado"""
        self.model_name = model_name
        self.config = config
        self.persona = persona
        self.model_dir = os.path.join("models", model_name)
        
        # Verificar se o modelo existe
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_dir}")
        
        # Carregar modelo e tokenizer
        try:
            self.model = LunaModel.from_pretrained(self.model_dir)
            self.tokenizer = LunaTokenizer.load(os.path.join(self.model_dir, "tokenizer"))
            
            # Detectar dispositivo
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.model.to(self.device)
            logger.info(f"Modelo carregado no dispositivo: {self.device}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            # Criar fallback mínimo para evitar falhas
            logger.warning("Usando modelo fallback devido a erro no carregamento do modelo original")
            self.fallback_mode = True
            self.model = None
            self.tokenizer = None
        else:
            self.fallback_mode = False
        
        logger.info(f"Chat inicializado com persona: {persona}")
    
    def chat(self, initial_context: str = ""):
        """Inicia uma sessão de chat interativa com o usuário"""
        print("\nBem-vindo ao chat com LunaGPT! (Digite 'sair' para terminar)")
        
        if initial_context:
            print(f"\nContexto inicial: {initial_context}")
        
        history = initial_context
        
        while True:
            user_input = input("\nVocê: ")
            
            if user_input.lower() in ["sair", "exit", "quit", "q"]:
                print("\nEncerrando chat. Até mais!")
                break
            
            # Atualizar histórico com entrada do usuário
            history = f"{history}\nUsuário: {user_input}"
            
            # Gerar resposta
            response = self.generate_response(history)
            
            # Atualizar histórico com resposta do modelo
            history = f"{history}\nLuna: {response}"
            
            # Exibir resposta
            print(f"\nLuna: {response}")
    
    def generate_response(self, prompt, max_length=100):
        """Gera resposta para um prompt"""
        try:
            # Pré-processamento do prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            
            # Se o tokenizer não tiver um pad_token_id, use o eos_token_id
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Configurar parâmetros de geração
            gen_kwargs = {
                "max_length": max_length + len(input_ids[0]),
                "min_length": 10,
                "do_sample": True,  # Usar amostragem
                "top_p": 0.92,
                "top_k": 50,
                "temperature": 0.85,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Gerar texto
            with torch.no_grad():
                output_sequences = self.model.generate(input_ids=input_ids, **gen_kwargs)
            
            # Decodificar e limpar a saída
            generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            # Remover o prompt de entrada do texto gerado
            response = generated_text[len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True)):].strip()
            
            # Aplicar formatação baseada em persona
            response = self._apply_persona(response)
            
            return response
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {str(e)}")
            logger.error(traceback.format_exc())  # Log do stack trace completo
            return "Desculpe, ocorreu um erro ao gerar a resposta."
    
    def _apply_persona_style(self, prompt: str) -> str:
        """Aplica o estilo da persona ao prompt"""
        # Personalizar prompt de acordo com a persona selecionada
        persona_prompts = {
            "tecnico": f"{prompt}\n[PERSONA: técnico, preciso, objetivo]",
            "casual": f"{prompt}\n[PERSONA: amigável, informal, conversacional]",
            "formal": f"{prompt}\n[PERSONA: formal, profissional, educado]",
        }
        
        return persona_prompts.get(self.persona, persona_prompts["casual"])
    
    def _extract_response(self, prompt: str, full_response: str) -> str:
        """Extrai apenas a parte da resposta gerada pelo modelo"""
        if full_response.startswith(prompt):
            # Se a resposta começa com o prompt, remover o prompt
            response = full_response[len(prompt):].strip()
        else:
            # Caso contrário, tentar encontrar o início da resposta
            if "Luna:" in full_response:
                # Extrair texto após "Luna:"
                parts = full_response.split("Luna:", 1)
                response = parts[1].strip()
            else:
                # Caso não encontre marcador, usar a resposta completa
                response = full_response.strip()
        
        return response