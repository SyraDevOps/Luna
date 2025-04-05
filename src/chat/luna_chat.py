import os
import torch
import re
import logging
import time
import threading
import traceback
from typing import List, Dict, Optional, Any, Union

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, PreTrainedTokenizerFast
from src.utils.logging_utils import logger
from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer
from src.chat.proactive_messenger import ProactiveMessenger  # Import the correct version

logger = logging.getLogger(__name__)

class LunaChat:
    """Interface de chat para interação com o modelo Luna"""
    
    def __init__(self, model_name: str, config, persona: str = "casual"):
        """Inicializa o chat com um modelo treinado"""
        self.model_name = model_name
        self.config = config
        self.persona = persona
        self.model_dir = os.path.join("models", model_name)
        self.conversation_history = []
        
        # Verificar se o modelo existe
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_dir}")
        
        # Carregar modelo e tokenizer
        try:
            self.model = LunaModel.from_pretrained(self.model_dir)
            self.tokenizer = LunaTokenizer.load(os.path.join(self.model_dir, "tokenizer"))
            
            # Detectar dispositivo
            self.device = self.model.to_appropriate_device()
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
        
        # Inicializar ProactiveMessenger com inactivity_threshold de 60 segundos
        self.proactive_messenger = ProactiveMessenger(self, inactivity_threshold=60)
        self.proactive_messenger.register_suggestion_callback(self._handle_proactive_suggestion)
        self.proactive_messenger.start_monitoring()

    def _handle_proactive_suggestion(self, suggestion):
        """Processa uma sugestão proativa"""
        print(f"\n[Sugestão]: {suggestion}")
        
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
            
            # Armazenar entrada no histórico de conversação
            self.conversation_history.append({"user": user_input})
            
            # Atualizar histórico com entrada do usuário
            history = f"{history}\nUsuário: {user_input}"
            
            # Informar ao ProactiveMessenger sobre atividade
            self.proactive_messenger.reset_activity_timer()
            
            # Gerar resposta
            response = self.generate_response(history)
            
            # Armazenar resposta no histórico de conversação
            if len(self.conversation_history) > 0:
                self.conversation_history[-1]["response"] = response
            
            # Atualizar histórico com resposta do modelo
            history = f"{history}\nLuna: {response}"
            
            # Exibir resposta
            print(f"\nLuna: {response}")
    
    def generate_response(self, prompt, max_length=100):
        """Gera resposta para um prompt"""
        if self.fallback_mode:
            return "Desculpe, estou operando em modo de contingência e não posso gerar respostas completas."
            
        try:
            # Aplicar estilo da persona ao prompt
            persona_prompt = self._apply_persona_style(prompt)
            
            # Pré-processamento do prompt
            inputs = self.tokenizer(persona_prompt, return_tensors="pt")
            
            # Verificar se inputs é um dicionário e tem a chave input_ids
            if isinstance(inputs, dict):
                if "input_ids" not in inputs:
                    raise ValueError("Tokenizer não retornou input_ids no dicionário")
                input_ids = inputs["input_ids"].to(self.device)
            else:
                input_ids = inputs.input_ids.to(self.device)
            
            # Se o tokenizer não tiver um pad_token_id, use o eos_token_id
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Configurar parâmetros de geração
            gen_kwargs = {
                "max_length": max_length + len(input_ids[0]),
                "min_length": 10,
                "do_sample": True,
                "top_p": 0.92,
                "top_k": 50,
                "temperature": 0.85,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Gerar texto
            with torch.no_grad():
                output_sequences = self.model.model.generate(input_ids=input_ids, **gen_kwargs)
            
            # Decodificar e limpar a saída
            generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            # Extrair apenas a resposta
            response = self._extract_response(persona_prompt, generated_text)
            
            return response
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {str(e)}")
            logger.error(traceback.format_exc())
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
                parts = full_response.split("Luna:", 1)
                response = parts[1].strip()
            else:
                # Caso não encontre marcador, usar a resposta completa
                response = full_response.strip()
        
        return response