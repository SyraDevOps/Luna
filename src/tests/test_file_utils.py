import unittest
import os
import tempfile
import json
from src.utils.file_utils import load_file, load_data_from_patterns, load_text_file, load_csv_file, load_json_file

class TestFileUtils(unittest.TestCase):
    """Testes para as utilidades de arquivo"""
    
    def setUp(self):
        """Configuração para os testes"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name
        
    def tearDown(self):
        """Limpeza após os testes"""
        self.temp_dir.cleanup()
        
    def test_load_text_file(self):
        """Testa o carregamento de arquivos de texto"""
        # Criar arquivo de texto
        text_path = os.path.join(self.test_dir, "test.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("Pergunta: O que é LunaGPT?\nResposta: É um sistema de diálogo neural.\n\nOutra linha de texto.")
            
        # Carregar e verificar
        data = load_text_file(text_path)
        self.assertEqual(len(data), 2)  # Duas entradas (um par Q&A e uma linha separada)
        self.assertTrue("Pergunta:" in data[0])
        self.assertTrue("Resposta:" in data[0])
        
    def test_load_csv_file(self):
        """Testa o carregamento de arquivos CSV"""
        # Criar arquivo CSV
        csv_path = os.path.join(self.test_dir, "test.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("pergunta,resposta\n")
            f.write("O que é LLM?,É um modelo de linguagem grande\n")
            f.write("Como funciona?,Através de redes neurais\n")
            
        # Carregar e verificar
        data = load_csv_file(csv_path)
        self.assertEqual(len(data), 2)
        self.assertTrue("Pergunta: O que é LLM?" in data[0])
        self.assertTrue("Resposta: É um modelo de linguagem grande" in data[0])
        
    def test_load_json_file(self):
        """Testa o carregamento de arquivos JSON"""
        # Criar arquivo JSON
        json_path = os.path.join(self.test_dir, "test.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([
                {"instruction": "Qual a diferença entre LLMs?", "output": "Os modelos diferem em arquitetura, tamanho e treinamento."}
            ], f, ensure_ascii=False)
            
        # Carregar e verificar
        data = load_json_file(json_path)
        self.assertEqual(len(data), 1)
        self.assertTrue("Pergunta: Qual a diferença entre LLMs?" in data[0])
        self.assertTrue("Resposta: Os modelos diferem em arquitetura, tamanho e treinamento." in data[0])
        
    def test_load_data_from_patterns(self):
        """Testa o carregamento de dados a partir de padrões"""
        # Criar arquivos de teste
        os.makedirs(os.path.join(self.test_dir, "subdir"), exist_ok=True)
        
        with open(os.path.join(self.test_dir, "file1.txt"), "w", encoding="utf-8") as f:
            f.write("Conteúdo do arquivo 1")
        
        with open(os.path.join(self.test_dir, "subdir", "file2.txt"), "w", encoding="utf-8") as f:
            f.write("Conteúdo do arquivo 2")
            
        # Testar carregamento com padrões - usar valid_ratio em vez de split_ratio
        patterns = [os.path.join(self.test_dir, "*.txt"), os.path.join(self.test_dir, "subdir", "*.txt")]
        train_data, valid_data = load_data_from_patterns(patterns, auto_split=True, valid_ratio=0.5)
        
        self.assertEqual(len(train_data) + len(valid_data), 2)
        self.assertEqual(len(train_data), 1)
        self.assertEqual(len(valid_data), 1)

if __name__ == "__main__":
    unittest.main()