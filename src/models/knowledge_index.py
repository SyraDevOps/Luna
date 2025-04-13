import logging
import os
from typing import List, Dict, Optional, Any
import json

logger = logging.getLogger(__name__)

# Verificar se LlamaIndex está disponível
try:
    from llama_index import Document, ServiceContext, VectorStoreIndex, SimpleDirectoryReader
    from llama_index.storage.storage_context import StorageContext
    from llama_index.vector_stores import SimpleVectorStore
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    logger.warning("LlamaIndex não disponível. Algumas funcionalidades avançadas de indexação não estarão disponíveis.")

class KnowledgeIndex:
    """Gerenciador de índice de conhecimento baseado em LlamaIndex"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.index_dir = os.path.join("models", model_name, "knowledge")
        os.makedirs(self.index_dir, exist_ok=True)
        
        self.index = None
        self.available = LLAMAINDEX_AVAILABLE
        
        if self.available:
            self._initialize_index()
    
    def _initialize_index(self):
        """Inicializa o índice LlamaIndex"""
        try:
            # Carregar índice existente ou criar novo
            index_file = os.path.join(self.index_dir, "index.json")
            storage_dir = os.path.join(self.index_dir, "storage")
            
            if os.path.exists(index_file):
                # Carregar índice existente
                from llama_index.storage.storage_context import StorageContext
                from llama_index.indices.vector_store.base import VectorStoreIndex
                from llama_index.indices.loading import load_index_from_storage
                
                storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
                self.index = load_index_from_storage(storage_context)
                logger.info(f"Índice de conhecimento carregado de {storage_dir}")
            else:
                # Criar novo índice
                vector_store = SimpleVectorStore()
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self.index = VectorStoreIndex([], storage_context=storage_context)
                logger.info("Novo índice de conhecimento criado")
                
                # Persistir
                self.index.storage_context.persist(persist_dir=storage_dir)
                
        except Exception as e:
            logger.error(f"Erro ao inicializar índice LlamaIndex: {str(e)}")
            self.available = False
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Adiciona documentos ao índice de conhecimento
        
        Args:
            documents: Lista de textos para indexar
            metadatas: Lista de metadados para documentos
        """
        if not self.available or not documents:
            return False
            
        try:
            # Converter para documentos LlamaIndex
            llama_docs = []
            for i, text in enumerate(documents):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                llama_docs.append(Document(text=text, metadata=metadata))
            
            # Adicionar ao índice
            self.index.insert_documents(llama_docs)
            
            # Persistir índice
            storage_dir = os.path.join(self.index_dir, "storage")
            self.index.storage_context.persist(persist_dir=storage_dir)
            
            logger.info(f"{len(documents)} documentos adicionados ao índice de conhecimento")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao adicionar documentos ao índice: {str(e)}")
            return False
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Realiza consulta no índice de conhecimento
        
        Args:
            query_text: Texto da consulta
            top_k: Número máximo de resultados
            
        Returns:
            Lista de resultados com texto e metadata
        """
        if not self.available:
            return []
            
        try:
            # Realizar consulta
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query_text)
            
            # Extrair resultados
            results = []
            for node in response.source_nodes:
                results.append({
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": node.score if hasattr(node, "score") else 0.0
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Erro ao consultar índice de conhecimento: {str(e)}")
            return []
            
    def import_from_files(self, directory: str, glob_pattern: str = "*.txt") -> int:
        """
        Importa documentos de arquivos para o índice de conhecimento
        
        Args:
            directory: Diretório contendo os arquivos
            glob_pattern: Padrão para filtrar arquivos
            
        Returns:
            Número de documentos importados
        """
        if not self.available:
            return 0
            
        try:
            # Carregar arquivos usando SimpleDirectoryReader
            reader = SimpleDirectoryReader(
                input_dir=directory,
                recursive=True,
                file_extractor={"*.pdf": "default", "*.docx": "default", "*.txt": "default"}
            )
            documents = reader.load_data()
            
            # Adicionar documentos ao índice
            if documents:
                self.index.insert_documents(documents)
                
                # Persistir índice
                storage_dir = os.path.join(self.index_dir, "storage")
                self.index.storage_context.persist(persist_dir=storage_dir)
                
                logger.info(f"{len(documents)} documentos importados de {directory}")
                return len(documents)
            
            return 0
            
        except Exception as e:
            logger.error(f"Erro ao importar documentos de {directory}: {str(e)}")
            return 0