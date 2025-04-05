from src.models.supervised_dataset import SupervisedDataset
from src.utils.logging_utils import logger
import unittest
from src.config.config import Config
from src.models.tokenizer import LunaTokenizer
from src.config.config import FeedbackConfig  # Exemplo no arquivo src/models/feedback_system.py
from src.config.config import TrainingConfig  # Exemplo no arquivo src/models/feedback_system.py
from src.config.config import ModelConfig  # Exemplo no arquivo src/models/tokenizer.py

class TestSupervisedDataset(unittest.TestCase):
    def setUp(self):
        self.config = Config()  # Inicialize corretamente a classe Config
        self.tokenizer = LunaTokenizer(self.config).tokenizer
        self.data = ["Teste de entrada", "Outro exemplo de teste"]

    def test_length(self):
        dataset = SupervisedDataset(self.tokenizer, self.data, max_length=64, augmentation=False)
        self.assertEqual(len(dataset), len(self.data))

    def test_item_keys(self):
        dataset = SupervisedDataset(self.tokenizer, self.data, max_length=64, augmentation=False)
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("labels", item)

if __name__ == "__main__":
    unittest.main()