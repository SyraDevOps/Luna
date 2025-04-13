import os
import logging
from typing import List, Optional
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

logger = logging.getLogger(__name__)

class LunaTokenizer:
    """Tokenizer especializado para o modelo Luna"""
    
    def __init__(self, config=None):
        """Inicializa o tokenizer Luna"""
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
    
    def augment_and_train(self, texts, output_dir):
        """Augmenta dados e treina o tokenizer"""
        augmented_texts = [advanced_augment(text) for text in texts]
        self.train_and_save(augmented_texts, output_dir)
    
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
    
    def load_from_directory(self, tokenizer_dir):
        """Carrega o tokenizer a partir de um diretório"""
        try:
            from transformers import PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
            return self
        except Exception as e:
            logger.error(f"Erro ao carregar tokenizer: {str(e)}")
            raise

    def load(self, tokenizer_dir):
        """Carrega o tokenizer"""
        return self.load_from_directory(tokenizer_dir)
    
    def __call__(self, *args, **kwargs):
        """Permite chamar o tokenizer diretamente"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer não inicializado")
        return self.tokenizer(*args, **kwargs)
        
    def encode(self, text, **kwargs):
        """Encapsula o método encode do tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer não inicializado")
        return self.tokenizer.encode(text, **kwargs)
        
    def decode(self, token_ids, **kwargs):
        """Encapsula o método decode do tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer não inicializado")
        return self.tokenizer.decode(token_ids, **kwargs)

    def convert_ids_to_tokens(self, ids, **kwargs):
        """Encapsula o método convert_ids_to_tokens"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer não inicializado")
        return self.tokenizer.convert_ids_to_tokens(ids, **kwargs)