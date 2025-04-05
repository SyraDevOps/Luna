import os
import logging
from typing import List, Optional
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers import ByteLevelBPETokenizer

logger = logging.getLogger(__name__)

class LunaTokenizer:
    """Tokenizer especializado para o modelo Luna"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
    
    def train_and_save(self, texts, output_dir):
        """Treina o tokenizer do zero e salva"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Criar um tokenizer BPE
        tokenizer_base = Tokenizer(models.BPE())
        tokenizer_base.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer_base.decoder = decoders.ByteLevel()
        
        # Definir os tokens especiais
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.model.vocab_size,
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[SEP]", "[CLS]", "[MASK]"]
        )
        
        # Treinar o tokenizer
        tokenizer_base.train_from_iterator(texts, trainer)
        
        # Converter para formato HuggingFace
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_base,
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token="[UNK]",
            sep_token="[SEP]",
            cls_token="[CLS]",
            mask_token="[MASK]"
        )
        
        # Salvar o tokenizer
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Tokenizer treinado e salvo em {output_dir}")
        
        return self.tokenizer
    
    def configure_special_tokens(self):
        """Configura os tokens especiais"""
        if self.tokenizer is None:
            logger.warning("Tokenizer não inicializado durante a configuração de tokens especiais")
            return
        
        # Garantir que os tokens especiais estejam configurados
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "[PAD]"
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = "[BOS]"
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "[EOS]"
        
        # Importante para GPT2: se não tiver pad_token_id, use eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info("Tokens especiais configurados corretamente")
    
    @classmethod
    def load(cls, tokenizer_path: str):
        """Carrega tokenizer de um diretório"""
        instance = cls(None)
        instance.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        instance.configure_special_tokens()
        logger.info(f"Tokenizer carregado de {tokenizer_path} com {instance.tokenizer.vocab_size} tokens")
        return instance

    def load(self, tokenizer_dir):
        """Carrega um tokenizer existente."""
        try:
            # Verificar se é um diretório de tokenizer válido
            if not os.path.exists(tokenizer_dir):
                logger.warning(f"Diretório do tokenizer não encontrado: {tokenizer_dir}")
                # Tentar criar um tokenizer vazio
                self.tokenizer = None
                return False
                
            # Verificar se há arquivos de tokenizer no formato HuggingFace
            if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
                self.configure_special_tokens()
                logger.info(f"Tokenizer carregado de {tokenizer_dir} com {len(self.tokenizer.get_vocab())} tokens")
                return True
                
            # Verificar arquivos no formato tokenizers
            vocab_file = os.path.join(tokenizer_dir, "vocab.json")
            merges_file = os.path.join(tokenizer_dir, "merges.txt")
            if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
                logger.error(f"Arquivos do tokenizer não encontrados em: {tokenizer_dir}")
                # Tentar criar um tokenizer vazio como fallback
                self.tokenizer = None
                return False
                
            # Carregar com ByteLevelBPETokenizer
            self.tokenizer = ByteLevelBPETokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file
            )
            self.configure_special_tokens()
            
            logger.info(f"Tokenizer carregado de {tokenizer_dir}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar tokenizer: {str(e)}")
            # Tokenizer vazio como fallback
            self.tokenizer = None
            return False

    def encode(self, text: str, **kwargs):
        """Codifica texto em tokens"""
        return self.tokenizer(text, **kwargs)
    
    def decode(self, token_ids, **kwargs):
        """Decodifica tokens em texto"""
        return self.tokenizer.decode(token_ids, **kwargs)