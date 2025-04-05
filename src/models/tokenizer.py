import os
import logging
from typing import List, Optional
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

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

    def load_from_directory(self, tokenizer_dir):
        """
        Carrega o tokenizer de um diretório específico.
        
        Args:
            tokenizer_dir: Caminho para o diretório contendo o tokenizer
        """
        try:
            from tokenizers import Tokenizer
            
            # Verificar se o diretório existe
            if not os.path.exists(tokenizer_dir):
                raise FileNotFoundError(f"Diretório do tokenizer não encontrado: {tokenizer_dir}")
            
            # Verificar arquivo do tokenizer
            tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
            if not os.path.exists(tokenizer_file):
                raise FileNotFoundError(f"Arquivo do tokenizer não encontrado: {tokenizer_file}")
                
            # Carregar o tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            
            # Configurar tokens especiais
            self.configure_special_tokens()
            
            logger.info(f"Tokenizer carregado com sucesso de {tokenizer_dir}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar tokenizer: {str(e)}")
            raise

    def encode(self, text: str, **kwargs):
        """Codifica texto em tokens"""
        return self.tokenizer(text, **kwargs)
    
    def decode(self, token_ids, **kwargs):
        """Decodifica tokens em texto"""
        return self.tokenizer.decode(token_ids, **kwargs)