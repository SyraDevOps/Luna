import unittest
import os
import tempfile
import shutil
from src.config.config import Config
from src.models.tokenizer import LunaTokenizer

class TestTokenizer(unittest.TestCase):
    """Testes para o módulo de tokenizer"""
    
    def setUp(self):
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
        self.config.model.tokenizer_path = os.path.join(self.temp_dir, "tokenizer")
        
        # Dados de amostra para treinar um tokenizer
        self.config.tokenizer_training_data = [
            "Este é um texto de teste em português.",
            "O tokenizer precisa funcionar bem com acentuação e caracteres especiais.",
            "Sistemas de processamento de linguagem natural são complexos."
        ]
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_tokenizer_creation(self):
        """Testa a criação de um novo tokenizer"""
        tokenizer = LunaTokenizer(self.config)
        texts = ["Este é um exemplo de texto para treinar o tokenizer."]
        
        # Usar diretório temporário para o teste
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Treinar e salvar o tokenizer
            tokenizer.train_and_save(texts, tmp_dir)
            
            # Verificar se o tokenizer foi criado
            self.assertIsNotNone(tokenizer.tokenizer)
    
    def test_tokenization_special_chars(self):
        """Testa tokenização de caracteres especiais"""
        # Inicializar o tokenizer primeiro se necessário
        tokenizer = LunaTokenizer(self.config)
        example_texts = ["Este é um exemplo de texto para treinar o tokenizer."]
        
        # Usar diretório temporário para o teste
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Treinar e salvar o tokenizer
            tokenizer.train_and_save(example_texts, tmp_dir)
            
            # Agora testar com caracteres especiais
            text = "Símbolos especiais: !@#$%^&*()_+{}|:<>?[];\',./`~"
            
            # Verificar se o tokenizer foi inicializado corretamente
            self.assertIsNotNone(tokenizer.tokenizer)
            
            # Obter tokens
            tokens = tokenizer.tokenizer(text)
            self.assertIsNotNone(tokens)