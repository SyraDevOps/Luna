import unittest
from src.utils.optional_dependencies import load_optional_dependencies

class TestOptionalDependencies(unittest.TestCase):
    def test_load_stanza(self):
        deps = load_optional_dependencies()
        self.assertIn('nlp', deps)
        self.assertIsNotNone(deps['nlp'])

if __name__ == "__main__":
    unittest.main()