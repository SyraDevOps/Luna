import os
import csv
import json
import glob
import random
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Importações para diferentes formatos de arquivo
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PyPDF2 não encontrado. Suporte a PDF desabilitado.")

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    logger.warning("python-docx não encontrado. Suporte a DOCX desabilitado.")

def load_data_from_patterns(patterns, auto_split=True, valid_ratio=0.1):
    """
    Carrega dados de arquivos que correspondem aos padrões especificados
    """
    data = []
    
    # Garantir que patterns seja uma lista
    if isinstance(patterns, str):
        patterns = [patterns]
        
    for pattern in patterns:
        files = glob.glob(pattern)
        logger.info(f"Encontrados {len(files)} arquivos em: {pattern}")
        
        # Processar cada arquivo encontrado
        for file in files:
            file_data = load_file(file)
            data.extend(file_data)
    
    logger.info(f"Total de {len(data)} amostras carregadas")
    
    # Dividir dados em treino e validação se necessário
    if auto_split and data and valid_ratio > 0:
        split_idx = int(len(data) * (1 - valid_ratio))
        # Embaralhar dados antes de dividir
        random.shuffle(data)
        return data[:split_idx], data[split_idx:]
    else:
        # Retornar todos os dados como treino e uma lista vazia como validação
        return data, []

def load_file(filepath):
    """Carrega o conteúdo de um arquivo baseado em sua extensão"""
    ext = Path(filepath).suffix.lower()
    
    try:
        if ext == '.txt':
            return load_text_file(filepath)
        elif ext == '.csv':
            return load_csv_file(filepath)
        elif ext == '.json' or ext == '.jsonl':
            return load_json_file(filepath)
        elif ext == '.pdf':
            return load_pdf_file(filepath)
        elif ext == '.docx':
            return load_docx_file(filepath)
        else:
            logger.warning(f"Formato de arquivo não suportado: {ext}")
            return []
    except Exception as e:
        logger.error(f"Erro ao processar {filepath}: {str(e)}")
        return []

def load_text_file(filepath):
    """Carrega dados de um arquivo de texto"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Detectar se há pares de pergunta/resposta
        qa_pairs = []
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Detectar padrão "Pergunta: ... Resposta: ..."
            if line.startswith("Pergunta:") and i + 1 < len(lines):
                if lines[i + 1].startswith("Resposta:"):
                    qa_pair = f"{line}\n{lines[i + 1]}"
                    qa_pairs.append(qa_pair)
                    i += 2
                    continue
            
            # Detectar padrão "Q: ... A: ..."
            if line.startswith("Q:") and i + 1 < len(lines):
                if lines[i + 1].startswith("A:"):
                    qa_pair = f"Pergunta: {line[2:].strip()}\nResposta: {lines[i + 1][2:].strip()}"
                    qa_pairs.append(qa_pair)
                    i += 2
                    continue
            
            # Linha individual
            if len(line) > 10:  # Filtrar linhas muito curtas
                qa_pairs.append(line)
            
            i += 1
        
        return qa_pairs

    except UnicodeDecodeError:
        # Tentar com encoding diferente
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
                return [content] if content.strip() else []
        except Exception as e:
            logger.error(f"Erro de encoding em {filepath}: {str(e)}")
            return []

def load_csv_file(filepath):
    """Carrega dados de um arquivo CSV"""
    qa_pairs = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Detectar delimitador
            sample = f.read(1024)
            f.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                # Tentar diferentes combinações de colunas
                question_keys = ['pergunta', 'question', 'prompt', 'input', 'texto']
                answer_keys = ['resposta', 'answer', 'response', 'output', 'target']
                
                question = None
                answer = None
                
                # Buscar pergunta
                for key in question_keys:
                    for col in row.keys():
                        if key.lower() in col.lower():
                            question = row[col]
                            break
                    if question:
                        break
                
                # Buscar resposta
                for key in answer_keys:
                    for col in row.keys():
                        if key.lower() in col.lower():
                            answer = row[col]
                            break
                    if answer:
                        break
                
                if question and answer:
                    qa_pair = f"Pergunta: {question.strip()}\nResposta: {answer.strip()}"
                    qa_pairs.append(qa_pair)
                elif question:
                    qa_pairs.append(question.strip())
        
        logger.info(f"Carregados {len(qa_pairs)} exemplos de {filepath}")
        return qa_pairs
        
    except Exception as e:
        logger.error(f"Erro ao carregar CSV {filepath}: {str(e)}")
        return []

def load_json_file(filepath):
    """Carrega dados de um arquivo JSON"""
    qa_pairs = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.jsonl'):
                # JSON Lines
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        qa_pairs.extend(extract_qa_from_json(data))
            else:
                # JSON regular
                data = json.load(f)
                qa_pairs.extend(extract_qa_from_json(data))
        
        logger.info(f"Carregados {len(qa_pairs)} exemplos de {filepath}")
        return qa_pairs
        
    except Exception as e:
        logger.error(f"Erro ao carregar JSON {filepath}: {str(e)}")
        return []

def extract_qa_from_json(data):
    """Extrai pares de pergunta/resposta de dados JSON"""
    qa_pairs = []
    
    if isinstance(data, dict):
        # Tentar diferentes estruturas
        if "conversations" in data:
            for conv in data["conversations"]:
                if "q" in conv and "a" in conv:
                    qa_pair = f"Pergunta: {conv['q']}\nResposta: {conv['a']}"
                    qa_pairs.append(qa_pair)
        elif "instruction" in data and "output" in data:
            qa_pair = f"Pergunta: {data['instruction']}\nResposta: {data['output']}"
            qa_pairs.append(qa_pair)
        elif "question" in data and "answer" in data:
            qa_pair = f"Pergunta: {data['question']}\nResposta: {data['answer']}"
            qa_pairs.append(qa_pair)
        elif "text" in data:
            qa_pairs.append(data["text"])
    elif isinstance(data, list):
        for item in data:
            qa_pairs.extend(extract_qa_from_json(item))
    
    return qa_pairs

def load_pdf_file(filepath):
    """Carrega texto de um arquivo PDF"""
    if not PDF_SUPPORT:
        logger.error("PyPDF2 não está instalado. Não é possível processar PDFs.")
        return []
    
    text_content = []
    try:
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():
                    # Processar texto para extrair Q&A
                    processed_text = process_pdf_text(text)
                    text_content.extend(processed_text)
        
        logger.info(f"Carregados {len(text_content)} exemplos de {filepath}")
        return text_content
        
    except Exception as e:
        logger.error(f"Erro ao carregar PDF {filepath}: {str(e)}")
        return []

def process_pdf_text(text):
    """Processa texto extraído de PDF para encontrar padrões Q&A"""
    qa_pairs = []
    
    # Dividir em linhas e limpar
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    current_qa = []
    for line in lines:
        # Detectar padrões de pergunta/resposta
        if any(pattern in line.lower() for pattern in ['pergunta:', 'questão:', 'q:', '?']):
            if current_qa:
                qa_pairs.append(' '.join(current_qa))
                current_qa = []
            current_qa.append(line)
        elif any(pattern in line.lower() for pattern in ['resposta:', 'r:', 'resp:']):
            current_qa.append(line)
        elif current_qa and len(line) > 20:
            current_qa.append(line)
        elif not current_qa and len(line) > 30:
            # Texto standalone
            qa_pairs.append(line)
    
    # Adicionar último Q&A se existir
    if current_qa:
        qa_pairs.append(' '.join(current_qa))
    
    return qa_pairs

def load_docx_file(filepath):
    """Carrega texto de um arquivo DOCX"""
    if not DOCX_SUPPORT:
        logger.error("python-docx não está instalado. Não é possível processar DOCX.")
        return []
    
    try:
        doc = Document(filepath)
        text_content = []
        
        current_text = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                current_text.append(text)
                
                # Se linha termina com ponto ou ?, considerar fim de seção
                if text.endswith(('.', '?', '!')):
                    if len(' '.join(current_text)) > 30:
                        text_content.append(' '.join(current_text))
                    current_text = []
        
        # Adicionar último texto se existir
        if current_text:
            text_content.append(' '.join(current_text))
        
        logger.info(f"Carregados {len(text_content)} exemplos de {filepath}")
        return text_content
        
    except Exception as e:
        logger.error(f"Erro ao carregar DOCX {filepath}: {str(e)}")
        return []

def save_data_to_file(data: List[str], filepath: str, format: str = "txt") -> bool:
    """
    Salva dados em um arquivo no formato especificado
    
    Args:
        data: Lista de strings para salvar
        filepath: Caminho do arquivo de saída
        format: Formato do arquivo ('txt', 'json', 'jsonl')
    
    Returns:
        True se salvou com sucesso, False caso contrário
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format == "txt":
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(f"{item}\n\n")
                    
        elif format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        elif format == "jsonl":
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    json.dump({"text": item}, f, ensure_ascii=False)
                    f.write('\n')
        
        logger.info(f"Dados salvos em {filepath} (formato: {format})")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao salvar dados em {filepath}: {str(e)}")
        return False

def get_file_hash(filepath: str) -> str:
    """Calcula hash MD5 de um arquivo"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""

def ensure_directory_exists(filepath: str):
    """Garante que o diretório do arquivo existe"""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

def clean_text(text: str) -> str:
    """Limpa e normaliza texto"""
    if not text:
        return ""
    
    # Remover espaços extras
    text = ' '.join(text.split())
    
    # Remover caracteres de controle
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    return text.strip()

def split_text_into_chunks(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
    """Divide texto em chunks menores com sobreposição"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # Tentar encontrar quebra natural (espaço, ponto)
        if end < len(text):
            # Procurar por ponto ou quebra de linha
            for break_char in ['. ', '\n', '? ', '! ']:
                break_pos = text.rfind(break_char, start, end)
                if break_pos > start:
                    end = break_pos + len(break_char)
                    break
            else:
                # Procurar por espaço
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Próximo início com overlap
        start = end - overlap if end - overlap > start else end
        
        # Evitar loop infinito
        if start >= len(text):
            break
    
    return chunks