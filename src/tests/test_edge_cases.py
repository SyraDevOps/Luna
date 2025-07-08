import unittest
from src.models.supervised_dataset import SupervisedDataset
from src.models.feedback_system import FeedbackSystem
from src.utils.exception_utils import log_exception
from src.utils.logging_utils import logger
from src.models.tokenizer import LunaTokenizer
from src.config.config import Config
from src.models.luna_model import advanced_augment, syntactic_reorder, OPTIONAL
import os

class TestEdgeCases(unittest.TestCase):
    def test_empty_text_syntactic_reorder(self):
        result = syntactic_reorder("")
        self.assertEqual(result, "")
    
    def test_empty_text_advanced_augment(self):
        result = advanced_augment("")
        self.assertEqual(result, "")
    
    def test_missing_stanza_dependency(self):
        backup = OPTIONAL.get('nlp')
        OPTIONAL['nlp'] = None
        result = syntactic_reorder("Teste de texto.")
        self.assertEqual(result, "Teste de texto.")
        OPTIONAL['nlp'] = backup

if __name__ == "__main__":
    unittest.main()