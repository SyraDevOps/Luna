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
from src.models.memory_system import MemorySystem
from src.models.adaptive_tokenizer import AdaptiveTokenizer
from src.utils.text_utils import clean_model_output

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
        
        # Registrar classes seguras para desserialização
        from src.models.luna_model import LunaModel
        from src.models.moe import MoEBlock
        from src.models.growing_network import StateSpaceLayer, GrowingNetwork
        from src.models.hypernet import HyperNetwork
        import torch.serialization
        torch.serialization.add_safe_globals([LunaModel, MoEBlock, StateSpaceLayer, 
                                             GrowingNetwork, HyperNetwork])
        
        # Carregar modelo e tokenizer
        try:
            self.model = LunaModel.from_pretrained(self.model_dir)
            tokenizer_instance = LunaTokenizer(self.config)
            self.tokenizer = tokenizer_instance.load(os.path.join(self.model_dir, "tokenizer"))
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
            
        # Inicializar sistema de memória
        self.memory = MemorySystem(model_name, config)
        
        # Detectar dispositivo
        self.device = self.model.to_appropriate_device()
        logger.info(f"Modelo carregado no dispositivo: {self.device}")
        
        logger.info(f"Chat inicializado com persona: {persona}")
        
        # Inicializar ProactiveMessenger com inactivity_threshold de 60 segundos
        self.proactive_messenger = ProactiveMessenger(self, inactivity_threshold=60)
        self.proactive_messenger.register_suggestion_callback(self._handle_proactive_suggestion)
        self.proactive_messenger.start_monitoring()
        
        # Inicializar tokenizer adaptativo
        self.adaptive_tokenizer = AdaptiveTokenizer(model_name, config, self.tokenizer.tokenizer)
        self.tokens_learning_active = getattr(config, "enable_tokens_learning", True)
        
        # Usar frequência definida na configuração (7 mensagens)
        self.tokens_learning_threshold = getattr(config, "tokens_update_frequency", 7)
        self.message_counter = 0

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
        if not prompt:
            return "Por favor, faça uma pergunta ou inicie uma conversa."
        
        try:
            # Analisar texto para tokenizer adaptativo
            if self.tokens_learning_active:
                self.adaptive_tokenizer.analyze_text(prompt)
                self.message_counter += 1
                
                # Verificar se é hora de adaptar o tokenizer
                if self.message_counter >= self.tokens_learning_threshold:
                    # Verificar se há candidatos suficientes
                    if self.adaptive_tokenizer.has_enough_candidates(3):
                        print("\n🔍 VOCABULÁRIO: Verificando candidatos a novos tokens...")
                        num_added = self.adaptive_tokenizer.extend_tokenizer(self.model)
                        if num_added > 0:
                            print(f"✅ VOCABULÁRIO: Adicionados {num_added} novos tokens ao vocabulário!")
                        self.message_counter = 0  # Reinicia apenas se processou
                    else:
                        print("\n⏳ VOCABULÁRIO: Poucos candidatos, adiando atualização...")
                        # Incrementar o limite para próxima verificação
                        self.tokens_learning_threshold += 3  # Esperar mais 3 mensagens
                        self.message_counter = 0  # Reinicia o contador
            
            # Recuperar contexto relevante da memória - com feedback visual
            memory_context = self.memory.generate_context_for_query(prompt)
            
            if memory_context:
                print("\n🔍 MEMÓRIA: Contexto relevante recuperado:")
                print("-------------------------------------------")
                print(memory_context)
                print("-------------------------------------------")
                styled_prompt = self._apply_persona_style(
                    f"[Contexto de memória relevante:\n{memory_context}\n]\n\n{prompt}"
                )
            else:
                print("\n🔍 MEMÓRIA: Nenhum contexto relevante encontrado para esta pergunta.")
                styled_prompt = self._apply_persona_style(prompt)
            
            # Preparar input - AQUI ESTÁ A CORREÇÃO
            inputs = self.tokenizer(
                styled_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Mover para o dispositivo correto
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Gerar resposta
            output = self.model.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                top_p=0.92,
                temperature=0.8,
                no_repeat_ngram_size=3
            )
            
            # Decodificar resposta
            full_response = self.tokenizer.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extrair apenas a resposta gerada
            raw_response = self._extract_response(styled_prompt, full_response)
            
            # Limpar a resposta gerada
            clean_response = clean_model_output(raw_response)
            
            # Processar a conversa para memória
            self.memory.process_conversation(prompt, clean_response)
            
            return clean_response
            
        except Exception as e:
            # Registrar erro, mas não usar fallback
            logger.error(f"Erro ao gerar resposta: {str(e)}")
            return f"Desculpe, ocorreu um erro ao processar sua mensagem. Detalhes: {str(e)}"
    
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
    
    # Método para salvar memórias antes de encerrar
    def close(self):
        """Salva memórias e libera recursos"""
        try:
            self.memory.save()
            logger.info("Sistema de memória salvo com sucesso")
        except Exception as e:
            logger.error(f"Erro ao salvar sistema de memória: {str(e)}")