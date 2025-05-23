from src.models.supervised_dataset import SupervisedDataset
from src.utils.logging_utils import logger
import unittest
from src.config.config import Config
from src.models.tokenizer import LunaTokenizer

class TestSupervisedDataset(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        # Criar tokenizer simples para teste
        tokenizer = LunaTokenizer(self.config)
        # Treinar com dados mínimos
        texts = ["Teste de texto para tokenizer."]
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.train_and_save(texts, tmp_dir)
            self.tokenizer = tokenizer.tokenizer
        
        self.data = ["Teste de entrada", "Outro exemplo de teste"]

    def test_length(self):
        # Remover argumento augmentation que não existe
        dataset = SupervisedDataset(self.data, self.tokenizer, max_length=64)
        self.assertEqual(len(dataset), len(self.data))

    def test_item_keys(self):
        # Remover argumento augmentation que não existe
        dataset = SupervisedDataset(self.data, self.tokenizer, max_length=64)
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("labels", item)

if __name__ == "__main__":
    unittest.main()