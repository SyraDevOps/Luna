import os
import json
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

logger = logging.getLogger(__name__)

class LunaTokenizer:
    """Tokenizer customizado para o LunaGPT com suporte a tokens adaptativos"""
    
    def __init__(self, config, tokenizer_path: Optional[str] = None):
        """
        Inicializa o tokenizer Luna
        
        Args:
            config: Configuração do modelo
            tokenizer_path: Caminho para tokenizer pré-treinado (opcional)
        """
        self.config = config
        if config is not None and hasattr(config, "model") and hasattr(config.model, "vocab_size"):
            self.vocab_size = config.model.vocab_size
        else:
            self.vocab_size = 32000
        self.tokenizer = None
        self.special_tokens = {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "sep_token": "<sep>",
            "cls_token": "<cls>",
            "mask_token": "<mask>"
        }
        
        # Tokens especiais do Luna
        self.luna_special_tokens = {
            "persona_start": "<persona>",
            "persona_end": "</persona>",
            "memory_start": "<memory>",
            "memory_end": "</memory>",
            "emotion_start": "<emotion>",
            "emotion_end": "</emotion>",
            "context_start": "<context>",
            "context_end": "</context>"
        }
        
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load_from_directory(tokenizer_path)
        else:
            logger.info("Tokenizer será treinado do zero quando necessário")
    
    def create_tokenizer(self) -> Tokenizer:
        """Cria um novo tokenizer BPE"""
        # Usar BPE (Byte Pair Encoding)
        tokenizer = Tokenizer(models.BPE(unk_token=self.special_tokens["unk_token"]))
        
        # Configurar pré-processamento
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # Configurar decodificador
        tokenizer.decoder = decoders.ByteLevel()
        
        # Configurar pós-processamento
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
        return tokenizer
    
    def train_and_save(self, texts: List[str], save_directory: str):
        """
        Treina o tokenizer com os textos fornecidos e salva
        
        Args:
            texts: Lista de textos para treinamento
            save_directory: Diretório onde salvar o tokenizer
        """
        logger.info(f"Treinando tokenizer com {len(texts)} textos")
        
        # Criar diretório se não existir
        os.makedirs(save_directory, exist_ok=True)
        
        # Preparar textos para treinamento
        if not texts:
            texts = ["Exemplo de texto para treinar o tokenizer do LunaGPT."]
            logger.warning("Nenhum texto fornecido, usando texto de exemplo")
        
        # Salvar textos em arquivo temporário
        temp_file = os.path.join(save_directory, "training_texts.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(f"{text}\n")
        
        try:
            # Criar tokenizer
            tokenizer = self.create_tokenizer()
            
            # Configurar trainer
            all_special_tokens = list(self.special_tokens.values()) + list(self.luna_special_tokens.values())
            
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=2,
                special_tokens=all_special_tokens,
                show_progress=True
            )
            
            # Treinar tokenizer
            tokenizer.train([temp_file], trainer)
            
            # Salvar tokenizer
            tokenizer_file = os.path.join(save_directory, "tokenizer.json")
            tokenizer.save(tokenizer_file)
            
            # Criar wrapper do HuggingFace
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                **self.special_tokens,
                **self.luna_special_tokens
            )
            
            # Configurar tokens especiais
            self.configure_special_tokens()
            
            # Salvar wrapper do HuggingFace
            self.tokenizer.save_pretrained(save_directory)
            
            # Salvar metadados
            metadata = {
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens,
                "luna_special_tokens": self.luna_special_tokens,
                "training_texts_count": len(texts)
            }
            
            metadata_file = os.path.join(save_directory, "tokenizer_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Limpar arquivo temporário
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            logger.info(f"Tokenizer treinado e salvo em {save_directory}")
            
        except Exception as e:
            logger.error(f"Erro ao treinar tokenizer: {e}")
            # Fallback: usar tokenizer pré-treinado
            self._create_fallback_tokenizer(save_directory)
    
    def _create_fallback_tokenizer(self, save_directory: str):
        """Cria tokenizer fallback baseado em GPT-2"""
        try:
            logger.info("Criando tokenizer fallback baseado em GPT-2")
            
            # Usar tokenizer do GPT-2 como base
            base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # Adicionar tokens especiais
            special_tokens_dict = {}
            if base_tokenizer.pad_token is None:
                special_tokens_dict["pad_token"] = self.special_tokens["pad_token"]
            if base_tokenizer.unk_token is None:
                special_tokens_dict["unk_token"] = self.special_tokens["unk_token"]
            if base_tokenizer.bos_token is None:
                special_tokens_dict["bos_token"] = self.special_tokens["bos_token"]
            if base_tokenizer.eos_token is None:
                special_tokens_dict["eos_token"] = self.special_tokens["eos_token"]
            
            # Adicionar tokens do Luna
            additional_tokens = list(self.luna_special_tokens.values())
            if additional_tokens:
                special_tokens_dict["additional_special_tokens"] = additional_tokens
            
            # Adicionar tokens especiais
            if special_tokens_dict:
                base_tokenizer.add_special_tokens(special_tokens_dict)
            
            self.tokenizer = base_tokenizer
            
            # Salvar tokenizer
            self.tokenizer.save_pretrained(save_directory)
            
            logger.info("Tokenizer fallback criado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao criar tokenizer fallback: {e}")
            raise
    
    def load_from_directory(self, tokenizer_directory: str):
        """
        Carrega tokenizer de um diretório
        
        Args:
            tokenizer_directory: Diretório contendo o tokenizer
        """
        try:
            if not os.path.exists(tokenizer_directory):
                logger.error(f"Diretório do tokenizer não encontrado: {tokenizer_directory}")
                return False
            
            # Tentar carregar tokenizer do HuggingFace
            tokenizer_config_path = os.path.join(tokenizer_directory, "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_directory)
                logger.info(f"Tokenizer carregado de {tokenizer_directory}")
                return True
            
            # Tentar carregar tokenizer.json
            tokenizer_json_path = os.path.join(tokenizer_directory, "tokenizer.json")
            if os.path.exists(tokenizer_json_path):
                tokenizer = Tokenizer.from_file(tokenizer_json_path)
                self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=tokenizer,
                    **self.special_tokens
                )
                logger.info(f"Tokenizer carregado de {tokenizer_json_path}")
                return True
            
            logger.error(f"Nenhum arquivo de tokenizer válido encontrado em {tokenizer_directory}")
            return False
            
        except Exception as e:
            logger.error(f"Erro ao carregar tokenizer: {e}")
            return False
    
    @classmethod
    def load(cls, tokenizer_directory: str, config=None):
        """
        Carrega o tokenizer a partir de um diretório
        
        Args:
            tokenizer_directory: Diretório contendo o tokenizer
            config: Configuração opcional
            
        Returns:
            Instância do LunaTokenizer
        """
        instance = cls(config)
        if instance.load_from_directory(tokenizer_directory):
            logger.info(f"Tokenizer carregado com sucesso de {tokenizer_directory}")
        else:
            logger.warning(f"Uso do tokenizer padrão, não foi possível carregar de {tokenizer_directory}")
        
        return instance
    
    def configure_special_tokens(self):
        """Configura tokens especiais no tokenizer"""
        if self.tokenizer is None:
            logger.error("Tokenizer não inicializado")
            return
        
        try:
            # Verificar e adicionar tokens especiais se necessário
            special_tokens_to_add = {}
            
            if self.tokenizer.pad_token is None:
                special_tokens_to_add["pad_token"] = self.special_tokens["pad_token"]
            if self.tokenizer.unk_token is None:
                special_tokens_to_add["unk_token"] = self.special_tokens["unk_token"]
            if self.tokenizer.bos_token is None:
                special_tokens_to_add["bos_token"] = self.special_tokens["bos_token"]
            if self.tokenizer.eos_token is None:
                special_tokens_to_add["eos_token"] = self.special_tokens["eos_token"]
            
            # Adicionar tokens do Luna
            luna_tokens = list(self.luna_special_tokens.values())
            existing_tokens = set(self.tokenizer.get_vocab().keys())
            new_luna_tokens = [token for token in luna_tokens if token not in existing_tokens]
            
            if new_luna_tokens:
                special_tokens_to_add["additional_special_tokens"] = new_luna_tokens
            
            # Adicionar tokens especiais
            if special_tokens_to_add:
                num_added = self.tokenizer.add_special_tokens(special_tokens_to_add)
                logger.info(f"Adicionados {num_added} tokens especiais")
            
            logger.info("Tokens especiais configurados")
            
        except Exception as e:
            logger.error(f"Erro ao configurar tokens especiais: {e}")
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Codifica texto em tokens
        
        Args:
            text: Texto para codificar
            **kwargs: Argumentos adicionais
            
        Returns:
            Lista de IDs de tokens
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer não inicializado")
        
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """
        Decodifica tokens em texto
        
        Args:
            token_ids: Lista de IDs de tokens
            **kwargs: Argumentos adicionais
            
        Returns:
            Texto decodificado
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer não inicializado")
        
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza texto em lista de tokens
        
        Args:
            text: Texto para tokenizar
            
        Returns:
            Lista de tokens
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer não inicializado")
        
        return self.tokenizer.tokenize(text)
    
    def get_vocab_size(self) -> int:
        """Retorna tamanho do vocabulário"""
        if self.tokenizer is None:
            return self.vocab_size
        return len(self.tokenizer.get_vocab())
    
    def get_vocab(self) -> Dict[str, int]:
        """Retorna o vocabulário completo"""
        if self.tokenizer is None:
            return {}
        return self.tokenizer.get_vocab()
    
    def add_tokens(self, new_tokens: List[str]) -> int:
        """
        Adiciona novos tokens ao vocabulário
        
        Args:
            new_tokens: Lista de novos tokens
            
        Returns:
            Número de tokens adicionados
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer não inicializado")
        
        return self.tokenizer.add_tokens(new_tokens)
    
    def save_vocabulary(self, save_directory: str):
        """
        Salva vocabulário em arquivo
        
        Args:
            save_directory: Diretório onde salvar
        """
        if self.tokenizer is None:
            logger.error("Tokenizer não inicializado")
            return
        
        try:
            os.makedirs(save_directory, exist_ok=True)
            
            vocab = self.get_vocab()
            vocab_file = os.path.join(save_directory, "vocab.json")
            
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Vocabulário salvo em {vocab_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar vocabulário: {e}")
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o tokenizer"""
        if self.tokenizer is None:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "vocab_size": self.get_vocab_size(),
            "model_type": type(self.tokenizer).__name__,
            "special_tokens": {
                "pad_token": self.tokenizer.pad_token,
                "unk_token": self.tokenizer.unk_token,
                "bos_token": self.tokenizer.bos_token,
                "eos_token": self.tokenizer.eos_token,
            },
            "luna_special_tokens": self.luna_special_tokens
        }
    
    def test_tokenizer(self, test_texts: Optional[List[str]] = None) -> bool:
        """
        Testa o tokenizer com textos de exemplo
        
        Args:
            test_texts: Textos para teste (opcional)
            
        Returns:
            True se todos os testes passaram
        """
        if self.tokenizer is None:
            logger.error("Tokenizer não inicializado")
            return False
        
        if test_texts is None:
            test_texts = [
                "Olá, como você está?",
                "Este é um teste do tokenizer do LunaGPT.",
                "Tokens especiais: <bos> <eos> <pad> <unk>",
                f"{self.luna_special_tokens['persona_start']} casual {self.luna_special_tokens['persona_end']}"
            ]
        
        try:
            for i, text in enumerate(test_texts):
                # Testar codificação
                tokens = self.encode(text)
                
                # Testar decodificação
                decoded = self.decode(tokens, skip_special_tokens=False)
                
                logger.info(f"Teste {i+1}: '{text}' -> {len(tokens)} tokens -> '{decoded}'")
            
            logger.info("Todos os testes do tokenizer passaram")
            return True
            
        except Exception as e:
            logger.error(f"Erro durante teste do tokenizer: {e}")
            return False