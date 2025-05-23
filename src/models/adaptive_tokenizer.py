import os
import json
import logging
import torch
import collections
from typing import List, Dict, Set, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AdaptiveTokenizer:
    """Sistema para adaptar dinamicamente o tokenizer com novos termos"""
    
    def __init__(self, model_name: str, config, base_tokenizer=None):
        """
        Inicializa o sistema de tokenizer adaptativo.
        """
        self.model_name = model_name
        self.config = config
        self.model_dir = os.path.join("models", model_name)
        
        # Configurações para adaptação de tokens
        self.min_frequency = getattr(config, "min_token_frequency", 5)
        self.max_new_tokens_per_session = getattr(config, "max_new_tokens_per_session", 1000000)
        self.unknown_terms_counter = collections.Counter()
        self.max_token_length = 25  # Tamanho máximo para um novo token
        
        # Arquivo para tokens candidatos durante treinamento
        self.token_collection_file = os.path.join(
            self.model_dir, 
            getattr(config, "token_collection_file", "token_candidates.json")
        )
        
        # Carregar ou inicializar o tokenizer
        self.tokenizer = base_tokenizer
        if self.tokenizer is None:
            from src.models.tokenizer import LunaTokenizer
            tokenizer_path = os.path.join(self.model_dir, "tokenizer")
            self.tokenizer = LunaTokenizer(config).load_from_directory(tokenizer_path)
        
        # Carregar termos já considerados
        self.considered_terms_file = os.path.join(self.model_dir, "considered_terms.json")
        self.considered_terms = self._load_considered_terms()
        
    def _load_considered_terms(self) -> Set[str]:
        """Carrega termos já considerados para evitar reconsideração"""
        if os.path.exists(self.considered_terms_file):
            try:
                with open(self.considered_terms_file, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.error(f"Erro ao carregar termos considerados: {str(e)}")
        
        return set()
        
    def _save_considered_terms(self):
        """Salva termos já considerados"""
        try:
            with open(self.considered_terms_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.considered_terms), f)
        except Exception as e:
            logger.error(f"Erro ao salvar termos considerados: {str(e)}")
    
    def analyze_text(self, text: str):
        """
        Analisa texto para encontrar termos desconhecidos ou subtokenizados
        
        Args:
            text: Texto a ser analisado
        """
        if not text.strip():
            return
            
        # Tokenizar o texto
        tokens = self.tokenizer(text)["input_ids"] if hasattr(self.tokenizer, "__call__") else self.tokenizer.encode(text)
        decoded_tokens = [self.tokenizer.decode([token]) for token in tokens]
        
        # Extrair palavras originais
        import re
        words = re.findall(r'\b[\w\'-]+\b', text)
        
        # Identificar termos que são subtokenizados
        for word in words:
            if len(word) < 3 or len(word) > self.max_token_length:
                continue  # Ignorar palavras muito curtas ou muito longas
                
            if word.lower() in self.considered_terms:
                continue  # Já considerado anteriormente
                
            # Tokenize esta palavra isoladamente
            word_tokens = self.tokenizer(word)["input_ids"] if hasattr(self.tokenizer, "__call__") else self.tokenizer.encode(word)
            
            # Se a palavra for dividida em múltiplos tokens, é um candidato
            if len(word_tokens) > 1:
                # Verificar se não é apenas um token especial + palavra
                decoded_word = self.tokenizer.decode(word_tokens)
                if word.strip() == decoded_word.strip():
                    self.unknown_terms_counter[word] += 1
                    
        # Registrar termos mais frequentes para debug
        if len(self.unknown_terms_counter) > 0:
            most_common = self.unknown_terms_counter.most_common(5)
            logger.debug(f"Termos desconhecidos mais frequentes: {most_common}")
    
    def get_candidate_tokens(self) -> List[str]:
        """
        Retorna lista de candidatos a novos tokens
        
        Returns:
            Lista de strings candidatas a tokens
        """
        candidates = []
        
        for term, count in self.unknown_terms_counter.items():
            if count >= self.min_frequency and term not in self.considered_terms:
                candidates.append(term)
                
        # Ordenar por frequência (mais frequente primeiro)
        candidates.sort(key=lambda x: self.unknown_terms_counter[x], reverse=True)
        
        # Limitar o número de novos tokens por sessão
        return candidates[:self.max_new_tokens_per_session]
    
    def has_enough_candidates(self, min_candidates=3):
        """Verifica se existem candidatos suficientes antes de processar"""
        candidates = 0
        for term, count in self.unknown_terms_counter.items():
            if count >= self.min_frequency and term not in self.considered_terms:
                candidates += 1
                if candidates >= min_candidates:
                    return True
        return False
    
    def extend_tokenizer(self, model=None) -> int:
        """
        Estende o tokenizer com novos tokens e atualiza os embeddings do modelo
        
        Args:
            model: Modelo para atualizar (opcional)
            
        Returns:
            Número de tokens adicionados
        """
        # Obter candidatos a tokens
        candidates = self.get_candidate_tokens()
        
        if not candidates:
            logger.info("Nenhum novo token para adicionar")
            return 0
            
        # Adicionar novos tokens ao tokenizer
        try:
            tokenizer = self.tokenizer.tokenizer if hasattr(self.tokenizer, 'tokenizer') else self.tokenizer
            original_vocab_size = len(tokenizer)
            
            # Adicionar tokens
            num_added = tokenizer.add_tokens(candidates)
            
            if num_added == 0:
                logger.info("Nenhum novo token foi adicionado (talvez já existam)")
                return 0
                
            logger.info(f"Adicionados {num_added} novos tokens ao tokenizer")
            
            # Atualizar o modelo se fornecido
            if model and hasattr(model, "model") and hasattr(model.model, "resize_token_embeddings"):
                logger.info(f"Atualizando embeddings do modelo para {len(tokenizer)} tokens")
                model.model.resize_token_embeddings(len(tokenizer))
                
                # Inicializar embeddings para os novos tokens com média dos subtoken embeddings
                if hasattr(model.model, "get_input_embeddings"):
                    logger.info("Inicializando embeddings para novos tokens")
                    embeddings = model.model.get_input_embeddings()
                    
                    for token in candidates:
                        # Obter ID do novo token
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        
                        # Obter subtokens e suas embeddings
                        subtoken_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
                        if len(subtoken_ids) > 0:
                            # Calcular a média das embeddings dos subtokens
                            with torch.no_grad():
                                subtoken_embeds = embeddings.weight[subtoken_ids]
                                avg_embedding = torch.mean(subtoken_embeds, dim=0)
                                # Atribuir ao novo token
                                embeddings.weight[token_id] = avg_embedding
                                
            # Salvar tokenizer atualizado
            tokenizer_dir = os.path.join(self.model_dir, "tokenizer")
            if hasattr(self.tokenizer, 'tokenizer'):
                self.tokenizer.tokenizer.save_pretrained(tokenizer_dir)
            else:
                self.tokenizer.save_pretrained(tokenizer_dir)
            
            # Atualizar lista de termos considerados
            self.considered_terms.update(candidates)
            self._save_considered_terms()
            
            # Limpar contador para a próxima sessão
            self.unknown_terms_counter.clear()
            
            return num_added
            
        except Exception as e:
            logger.error(f"Erro ao estender tokenizer: {str(e)}")
            return 0