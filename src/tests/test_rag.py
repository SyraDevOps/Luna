import unittest
import os
import numpy as np
import tempfile
import json
from unittest.mock import patch, MagicMock

from src.models.rag_retriever import RAGRetriever

class TestRAGRetriever(unittest.TestCase):
    """Testes para o sistema de RAG (Retrieval-Augmented Generation)"""
    
    def test_retriever_initialization(self):
        """Teste de inicialização do RAG Retriever"""
        # Inicializar com dimensões padrão
        retriever = RAGRetriever(embedding_dim=768)
        self.assertEqual(len(retriever.documents), 0)
        
        # Inicialização com dimensões personalizadas
        retriever = RAGRetriever(embedding_dim=128)
        self.assertEqual(retriever.embedding_dim, 128)
    
    @patch('src.models.rag_retriever.FAISS_AVAILABLE', False)
    def test_fallback_to_numpy(self):
        """Testa o fallback para numpy quando FAISS não está disponível"""
        retriever = RAGRetriever(embedding_dim=64)
        self.assertFalse(retriever.use_faiss)
        self.assertEqual(retriever.embeddings.shape, (0, 64))
    
    def test_add_and_retrieve_documents(self):
        """Testa adição de documentos e recuperação"""
        # Create a mock embedding function
        with patch.object(RAGRetriever, '_compute_embedding') as mock_embed:
            # Configurar retornos específicos para o mock
            mock_embed.side_effect = [
                np.ones(64, dtype=np.float32) * 0.8,  # Doc1 sobre IA tem a pontuação mais alta
                np.ones(64, dtype=np.float32) * 0.2,
                np.ones(64, dtype=np.float32) * 0.1,
                np.ones(64, dtype=np.float32) * 0.7,  # Query embedding (para recuperar doc1)
            ]
            
            # Create retriever and add documents
            retriever = RAGRetriever(embedding_dim=64)
            
            # Usar documentos onde o primeiro contém "IA"
            retriever.add_documents([
                "Este é um documento sobre IA avançada.",  # Este deve ser retornado primeiro
                "Este é um documento sobre processamento de linguagem.",
                "Este é um documento sobre aprendizado de máquina."
            ])
            
            # Retrieve documents
            results = retriever.retrieve("IA", top_k=2)
            
            # Verificações
            self.assertEqual(len(results), 2, "Deve retornar 2 documentos")
            self.assertIn("IA", results[0]["document"], "Primeiro documento deve conter 'IA'")
    
    def test_save_and_load(self):
        """Testa salvamento e carregamento do retriever"""
        retriever = RAGRetriever(embedding_dim=32)
        
        # Adicionar alguns documentos com embeddings simulados
        documents = ["Doc1", "Doc2", "Doc3"]
        metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        with patch.object(retriever, '_compute_embedding') as mock_embed:
            # Embeddings simulados
            mock_embed.side_effect = [
                np.ones(32, dtype=np.float32) * 0.1,
                np.ones(32, dtype=np.float32) * 0.2,
                np.ones(32, dtype=np.float32) * 0.3,
            ]
            
            retriever.add_documents(documents, metadatas)
        
        # Salvar em diretório temporário
        with tempfile.TemporaryDirectory() as tmpdirname:
            retriever.save(tmpdirname)
            
            # Verificar arquivos salvos
            self.assertTrue(os.path.exists(os.path.join(tmpdirname, "documents.json")))
            
            # Carregar de volta
            loaded_retriever = RAGRetriever.load(tmpdirname)
            
            # Verificar dados carregados
            self.assertEqual(len(loaded_retriever.documents), 3)
            self.assertEqual(loaded_retriever.documents, documents)
            self.assertEqual(loaded_retriever.document_metadatas, metadatas)

if __name__ == '__main__':
    unittest.main()