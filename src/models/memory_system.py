import os
import json
import torch
import logging
import numpy as np
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field

# Imports condicionais para os sistemas de vetores
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS não disponível. Usando fallback NumPy para armazenamento de vetores.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers não disponível. Usando embeddings simplificados.")

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Representação de uma entrada na memória"""
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    source: str = "conversation"  # conversation, knowledge, feedback
    importance: float = 1.0
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável"""
        result = {
            "content": self.content,
            "timestamp": self.timestamp,
            "source": self.source,
            "importance": self.importance,
            "metadata": self.metadata
        }
        # Não incluir embedding no dict para serialização JSON
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Cria entrada de memória a partir de um dicionário"""
        return cls(
            content=data["content"],
            timestamp=data["timestamp"],
            source=data["source"],
            importance=data["importance"],
            metadata=data.get("metadata", {})
        )

class MemorySystem:
    """Sistema unificado de memória para o LunaGPT"""
    
    def __init__(self, model_name: str, config=None):
        self.model_name = model_name
        self.config = config
        
        # Diretórios para armazenamento persistente
        self.memory_dir = os.path.join("models", model_name, "memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Arquivos de armazenamento
        self.episodic_file = os.path.join(self.memory_dir, "episodic_memory.jsonl")
        self.semantic_file = os.path.join(self.memory_dir, "semantic_memory.jsonl")
        self.index_file = os.path.join(self.memory_dir, "memory_index.faiss")
        
        # Memórias em RAM
        self.episodic_memory: List[MemoryEntry] = []
        self.semantic_memory: List[MemoryEntry] = []
        
        # Embedding model
        self.embedding_model = self._load_embedding_model()
        self.embedding_dim = 768  # Padrão, será ajustado ao carregar o modelo
        
        # Índice vetorial
        self.use_faiss = FAISS_AVAILABLE
        self.index = None
        self.embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        # Carregar memória existente
        self._init_memory_system()
        
    def _load_embedding_model(self):
        """Carrega modelo de embeddings"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Usar modelo multilíngue para melhor suporte a diferentes idiomas
                model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
                self.embedding_dim = model.get_sentence_embedding_dimension()
                logger.info(f"Modelo de embeddings carregado com dimensão {self.embedding_dim}")
                return model
            except Exception as e:
                logger.warning(f"Erro ao carregar modelo de embeddings: {str(e)}")
        return None
        
    def _init_memory_system(self):
        """Inicializa o sistema de memória"""
        # Criar índice FAISS ou fallback NumPy
        if self.use_faiss:
            try:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info(f"Índice FAISS inicializado com dimensão {self.embedding_dim}")
            except Exception as e:
                logger.warning(f"Erro ao inicializar FAISS: {str(e)}. Usando fallback.")
                self.use_faiss = False
                
        # Carregar memórias persistentes
        self._load_memories()
        
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Calcula embedding para texto"""
        if self.embedding_model is not None:
            try:
                return self.embedding_model.encode([text])[0].astype(np.float32)
            except Exception as e:
                logger.error(f"Erro ao calcular embedding: {str(e)}")
                
        # Fallback simples se não tiver modelo
        import hashlib
        # Usar hash do texto para criar um vetor pseudoaleatório mas determinístico
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        # Converter bytes para array e normalizar
        vector = np.array([float(b) for b in hash_bytes], dtype=np.float32)
        # Repetir para atingir dimensão desejada
        vector = np.tile(vector, 1 + self.embedding_dim // len(vector))[:self.embedding_dim]
        # Normalizar
        return vector / np.linalg.norm(vector)
                
    def _load_memories(self):
        """Carrega memórias de arquivos persistentes"""
        # Carregar memória episódica
        if os.path.exists(self.episodic_file):
            try:
                with open(self.episodic_file, 'r', encoding='utf-8') as f:
                    self.episodic_memory = [
                        MemoryEntry.from_dict(json.loads(line)) 
                        for line in f if line.strip()
                    ]
                logger.info(f"Carregadas {len(self.episodic_memory)} entradas de memória episódica")
            except Exception as e:
                logger.error(f"Erro ao carregar memória episódica: {str(e)}")
                self.episodic_memory = []
        
        # Carregar memória semântica
        if os.path.exists(self.semantic_file):
            try:
                with open(self.semantic_file, 'r', encoding='utf-8') as f:
                    self.semantic_memory = [
                        MemoryEntry.from_dict(json.loads(line)) 
                        for line in f if line.strip()
                    ]
                logger.info(f"Carregadas {len(self.semantic_memory)} entradas de memória semântica")
            except Exception as e:
                logger.error(f"Erro ao carregar memória semântica: {str(e)}")
                self.semantic_memory = []
                
        # Carregar índice de vetores
        self._load_vector_index()
        
    def _load_vector_index(self):
        """Carrega índice vetorial"""
        embeddings_file = os.path.join(self.memory_dir, "embeddings.npy")
        
        if self.use_faiss and os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
                logger.info(f"Índice FAISS carregado com {self.index.ntotal} vetores")
            except Exception as e:
                logger.error(f"Erro ao carregar índice FAISS: {str(e)}")
                self.use_faiss = False
        
        # Fallback para NumPy ou cálculo de novos embeddings
        if not self.use_faiss or not os.path.exists(self.index_file):
            if os.path.exists(embeddings_file):
                try:
                    self.embeddings = np.load(embeddings_file)
                    logger.info(f"Embeddings carregados do arquivo. Shape: {self.embeddings.shape}")
                except Exception as e:
                    logger.error(f"Erro ao carregar embeddings: {str(e)}")
                    # Recalcular embeddings para todas as memórias
                    self._recalculate_all_embeddings()
            else:
                # Se não há arquivo de embeddings, calcular todos
                self._recalculate_all_embeddings()
                
    def _recalculate_all_embeddings(self):
        """Recalcula embeddings para todas as memórias"""
        all_memories = self.episodic_memory + self.semantic_memory
        if not all_memories:
            return
            
        logger.info(f"Recalculando embeddings para {len(all_memories)} entradas de memória")
        
        # Calcular embeddings
        all_embeddings = np.vstack([
            self._compute_embedding(memory.content) 
            for memory in all_memories
        ])
        
        # Armazenar embeddings
        if self.use_faiss:
            try:
                if self.index.ntotal > 0:
                    self.index.reset()
                self.index.add(all_embeddings)
            except Exception as e:
                logger.error(f"Erro ao adicionar embeddings ao FAISS: {str(e)}")
                self.use_faiss = False
                self.embeddings = all_embeddings
        else:
            self.embeddings = all_embeddings
            
        # Atualizar embeddings nas entradas de memória
        for i, memory in enumerate(all_memories):
            memory.embedding = all_embeddings[i]
            
        logger.info("Recálculo de embeddings concluído")
    
    def add_episodic_memory(self, content: str, metadata: Dict = None) -> None:
        """
        Adiciona uma entrada à memória episódica
        
        Args:
            content: Conteúdo da interação
            metadata: Metadados associados (como timestamp, usuário, etc.)
        """
        if not content.strip():
            return
            
        # Criar entrada de memória
        metadata = metadata or {}
        memory = MemoryEntry(
            content=content,
            source="conversation",
            metadata=metadata
        )
        
        # Calcular embedding
        embedding = self._compute_embedding(content)
        memory.embedding = embedding
        
        # Adicionar à memória episódica
        self.episodic_memory.append(memory)
        
        # Adicionar ao índice vetorial
        self._add_embedding_to_index(embedding)
        
        # Salvar ao arquivo
        self._append_to_file(self.episodic_file, memory)
        
        logger.debug(f"Memória episódica adicionada: {content[:50]}...")
        
    def add_semantic_memory(self, content: str, importance: float = 1.0, metadata: Dict = None) -> None:
        """
        Adiciona um fato ou conhecimento à memória semântica
        
        Args:
            content: Conteúdo do conhecimento
            importance: Pontuação de importância (maior = mais relevante)
            metadata: Metadados associados
        """
        if not content.strip():
            return
            
        # Evitar duplicatas
        if any(mem.content == content for mem in self.semantic_memory):
            return
            
        # Criar entrada de memória
        metadata = metadata or {}
        memory = MemoryEntry(
            content=content,
            source="knowledge",
            importance=importance,
            metadata=metadata
        )
        
        # Calcular embedding
        embedding = self._compute_embedding(content)
        memory.embedding = embedding
        
        # Adicionar à memória semântica
        self.semantic_memory.append(memory)
        
        # Adicionar ao índice vetorial
        self._add_embedding_to_index(embedding)
        
        # Salvar ao arquivo
        self._append_to_file(self.semantic_file, memory)
        
        logger.debug(f"Memória semântica adicionada: {content[:50]}...")
        
    def _add_embedding_to_index(self, embedding: np.ndarray) -> None:
        """Adiciona embedding ao índice vetorial"""
        if self.use_faiss:
            try:
                self.index.add(np.array([embedding], dtype=np.float32))
            except Exception as e:
                logger.error(f"Erro ao adicionar ao índice FAISS: {str(e)}")
        else:
            # Fallback para NumPy
            if len(self.embeddings) == 0:
                self.embeddings = np.array([embedding], dtype=np.float32)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
                
    def _append_to_file(self, filepath: str, memory: MemoryEntry) -> None:
        """Salva entrada de memória em arquivo persistente"""
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(memory.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Erro ao salvar memória em {filepath}: {str(e)}")
            
    def save(self) -> None:
        """Salva todas as memórias e índices em armazenamento persistente"""
        # Salvar memória episódica
        try:
            with open(self.episodic_file, 'w', encoding='utf-8') as f:
                for memory in self.episodic_memory:
                    f.write(json.dumps(memory.to_dict()) + '\n')
            logger.info(f"Memória episódica salva com {len(self.episodic_memory)} entradas")
        except Exception as e:
            logger.error(f"Erro ao salvar memória episódica: {str(e)}")
            
        # Salvar memória semântica
        try:
            with open(self.semantic_file, 'w', encoding='utf-8') as f:
                for memory in self.semantic_memory:
                    f.write(json.dumps(memory.to_dict()) + '\n')
            logger.info(f"Memória semântica salva com {len(self.semantic_memory)} entradas")
        except Exception as e:
            logger.error(f"Erro ao salvar memória semântica: {str(e)}")
            
        # Salvar índice vetorial
        self._save_vector_index()
        
    def _save_vector_index(self) -> None:
        """Salva índice vetorial"""
        if self.use_faiss and self.index:
            try:
                faiss.write_index(self.index, self.index_file)
                logger.info(f"Índice FAISS salvo com {self.index.ntotal} vetores")
            except Exception as e:
                logger.error(f"Erro ao salvar índice FAISS: {str(e)}")
        else:
            # Fallback para NumPy
            embeddings_file = os.path.join(self.memory_dir, "embeddings.npy")
            try:
                np.save(embeddings_file, self.embeddings)
                logger.info(f"Embeddings salvos com shape {self.embeddings.shape}")
            except Exception as e:
                logger.error(f"Erro ao salvar embeddings: {str(e)}")
                
    def retrieve_relevant_memories(self, query: str, top_k: int = 5, 
                                  include_episodic: bool = True, 
                                  include_semantic: bool = True) -> List[MemoryEntry]:
        """
        Recupera memórias relevantes para uma consulta
        
        Args:
            query: Texto da consulta
            top_k: Número de memórias a retornar
            include_episodic: Se True, inclui memória episódica
            include_semantic: Se True, inclui memória semântica
            
        Returns:
            Lista de entradas de memória relevantes
        """
        if not query.strip():
            return []
            
        if not include_episodic and not include_semantic:
            return []
            
        # Obter todas as memórias candidatas
        memories = []
        if include_episodic:
            memories.extend(self.episodic_memory)
        if include_semantic:
            memories.extend(self.semantic_memory)
            
        if not memories:
            return []
            
        # Calcular embedding da consulta
        query_embedding = self._compute_embedding(query)
        
        # Recuperar memórias mais similares
        if self.use_faiss and self.index:
            try:
                distances, indices = self.index.search(np.array([query_embedding]), min(top_k, len(memories)))
                result = [memories[idx] for idx in indices[0] if idx >= 0 and idx < len(memories)]
                return result
            except Exception as e:
                logger.error(f"Erro ao pesquisar no índice FAISS: {str(e)}")
                # Fallback para pesquisa linear
        
        # Pesquisa linear (fallback)
        all_embeddings = np.vstack([memory.embedding if memory.embedding is not None 
                                   else self._compute_embedding(memory.content)
                                   for memory in memories])
        
        # Calcular similaridades
        similarities = np.dot(all_embeddings, query_embedding)
        
        # Ordenar por similaridade
        indices = np.argsort(similarities)[::-1][:min(top_k, len(memories))]
        
        # Retornar memórias mais similares
        return [memories[idx] for idx in indices]
        
    def extract_knowledge_from_conversation(self, conversation: str) -> List[str]:
        """
        Extrai fatos e conhecimentos de uma conversa para memória semântica
        
        Args:
            conversation: Texto da conversa
            
        Returns:
            Lista de fatos extraídos
        """
        # Implementação com tratamento de erro melhorado para NLTK
        try:
            import nltk
            # Baixar o recurso correto - é 'punkt', não 'punkt_tab'
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Baixando recursos NLTK necessários...")
                nltk.download('punkt', quiet=True)
            
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(conversation)
            print(f"MEMÓRIA: Processando {len(sentences)} sentenças para extração de conhecimento")
        except Exception as e:
            # Fallback para split simples em caso de erro com NLTK
            print(f"MEMÓRIA: Usando método simples de tokenização (erro NLTK: {str(e)})")
            sentences = [s.strip() + "." for s in conversation.split(".") if s.strip()]
        
        # Filtrar para sentenças que parecem declarativas e não perguntas
        facts = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and not sent.endswith('?'):
                if any(keyword in sent.lower() for keyword in 
                      ['é', 'são', 'possui', 'contém', 'significa', 'representa',
                       'consiste', 'implica', 'define', 'caracteriza']):
                    facts.append(sent)
        
        if facts:
            print(f"MEMÓRIA: Extraídos {len(facts)} fatos da conversa")
        
        return facts
        
    def generate_context_for_query(self, query: str, max_tokens: int = 1024) -> str:
        """
        Gera um contexto de memórias relevantes para uma consulta
        
        Args:
            query: Consulta do usuário
            max_tokens: Número máximo de tokens para o contexto
            
        Returns:
            String com contexto relevante
        """
        # Recuperar memórias relevantes
        relevant_memories = self.retrieve_relevant_memories(query, top_k=5)
        
        if not relevant_memories:
            return ""
            
        # Formatar contexto
        context_parts = []
        token_count = 0
        
        for memory in relevant_memories:
            # Estimativa simples de tokens (aproximadamente 4 caracteres por token)
            estimated_tokens = len(memory.content) // 4
            
            if token_count + estimated_tokens > max_tokens:
                break
                
            source_type = "Conversa anterior" if memory.source == "conversation" else "Conhecimento"
            
            context_parts.append(f"{source_type}: {memory.content}")
            token_count += estimated_tokens
            
        return "\n\n".join(context_parts)
        
    def process_conversation(self, user_input: str, assistant_response: str, 
                           conversation_id: str = None) -> None:
        """
        Processa uma conversa completa para memória
        
        Args:
            user_input: Entrada do usuário
            assistant_response: Resposta do assistente
            conversation_id: ID opcional da conversa
        """
        # Adicionar à memória episódica (conversa completa)
        conversation = f"Usuário: {user_input}\nAssistente: {assistant_response}"
        
        metadata = {
            "conversation_id": conversation_id or datetime.now().strftime("%Y%m%d%H%M%S"),
            "user_input": user_input,
            "assistant_response": assistant_response
        }
        
        # Mostrar mensagem clara no terminal
        print("\n💾 MEMÓRIA: Salvando conversa na memória episódica...")
        self.add_episodic_memory(conversation, metadata)
        
        # Extrair conhecimento para memória semântica
        facts = self.extract_knowledge_from_conversation(conversation)
        if facts:
            print(f"📚 MEMÓRIA: Salvando {len(facts)} fatos na memória semântica")
            for fact in facts:
                self.add_semantic_memory(fact)
            
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas sobre o sistema de memória"""
        return {
            "episodic_count": len(self.episodic_memory),
            "semantic_count": len(self.semantic_memory),
            "total_memories": len(self.episodic_memory) + len(self.semantic_memory),
            "index_type": "FAISS" if self.use_faiss else "NumPy",
            "embedding_dimension": self.embedding_dim
        }