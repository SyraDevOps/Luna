import os
import json
import csv
import re
import logging
from pathlib import Path

import glob
import random

logger = logging.getLogger(__name__)

# Importações para diferentes formatos de arquivo
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PyPDF2 não encontrado. Suporte a PDF desabilitado.")

try:
    import docx
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
            if PDF_SUPPORT:
                return load_pdf_file(filepath)
            else:
                logger.warning(f"Suporte a PDF não disponível. Instale PyPDF2: pip install PyPDF2")
                return []
        elif ext == '.docx':
            if DOCX_SUPPORT:
                return load_docx_file(filepath)
            else:
                logger.warning(f"Suporte a DOCX não disponível. Instale python-docx: pip install python-docx")
                return []
        else:
            logger.warning(f"Formato de arquivo não suportado: {ext} em {filepath}")
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
            if lines[i].startswith("Pergunta:") or lines[i].endswith("?"):
                if i+1 < len(lines) and (lines[i+1].startswith("Resposta:") or not lines[i+1].endswith("?")):
                    q = lines[i].replace("Pergunta:", "").strip()
                    a = lines[i+1].replace("Resposta:", "").strip()
                    qa_pairs.append(f"Pergunta: {q}\nResposta: {a}")
                    i += 2
                else:
                    qa_pairs.append(lines[i])
                    i += 1
            else:
                qa_pairs.append(lines[i])
                i += 1
                
        logger.info(f"Carregados {len(qa_pairs)} exemplos de {filepath}")
        return qa_pairs
    except UnicodeDecodeError:
        # Tentar outros encodings se utf-8 falhar
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"Arquivo {filepath} carregado com encoding {encoding}")
                return lines
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Não foi possível determinar o encoding de {filepath}")
        return []

def load_csv_file(filepath):
    """Carrega dados de um arquivo CSV"""
    qa_pairs = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Detectar o delimitador
            sample = f.readline()
            f.seek(0)
            
            if ',' in sample:
                delimiter = ','
            elif ';' in sample:
                delimiter = ';'
            elif '\t' in sample:
                delimiter = '\t'
            else:
                delimiter = ','
                
            reader = csv.reader(f, delimiter=delimiter)
            headers = next(reader, None)
            
            # Se não há cabeçalhos, usamos os índices
            if not headers:
                return []
                
            # Identificar colunas potenciais para pergunta/resposta
            q_col = -1
            a_col = -1
            
            for i, header in enumerate(headers):
                header_lower = header.lower()
                if any(term in header_lower for term in ['pergunta', 'questão', 'query', 'question', 'prompt', 'input']):
                    q_col = i
                elif any(term in header_lower for term in ['resposta', 'answer', 'response', 'reply', 'output']):
                    a_col = i
            
            # Se encontrou colunas de pergunta e resposta
            if q_col >= 0 and a_col >= 0:
                for row in reader:
                    if len(row) > max(q_col, a_col):
                        qa_pairs.append(f"Pergunta: {row[q_col].strip()}\nResposta: {row[a_col].strip()}")
            else:
                # Tentar usar as duas primeiras colunas
                for row in reader:
                    if len(row) >= 2:
                        qa_pairs.append(f"Pergunta: {row[0].strip()}\nResposta: {row[1].strip()}")
                    elif row:
                        qa_pairs.append(row[0].strip())
        
        logger.info(f"Carregados {len(qa_pairs)} exemplos de {filepath}")
        return qa_pairs
    except Exception as e:
        logger.error(f"Erro ao processar arquivo CSV {filepath}: {str(e)}")
        return []

def load_json_file(filepath):
    """Carrega dados de um arquivo JSON"""
    qa_pairs = []
    
    try:
        # Verificar se é um JSONL (uma linha JSON por linha)
        if filepath.lower().endswith('.jsonl'):
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if isinstance(item, dict):
                                # Procurar por campos de pergunta/resposta
                                q_field = next((k for k in item if any(term in k.lower() for term in 
                                             ['pergunta', 'questão', 'query', 'question', 'prompt', 'input'])), None)
                                a_field = next((k for k in item if any(term in k.lower() for term in 
                                             ['resposta', 'answer', 'response', 'reply', 'output'])), None)
                                
                                if q_field and a_field:
                                    qa_pairs.append(f"Pergunta: {item[q_field]}\nResposta: {item[a_field]}")
                                else:
                                    # Sem campos específicos, usar o texto completo
                                    qa_pairs.append(json.dumps(item, ensure_ascii=False))
                        except json.JSONDecodeError:
                            continue
        else:
            # JSON regular
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Processar diferentes estruturas JSON
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # Procurar por campos de pergunta/resposta
                            q_field = next((k for k in item if any(term in k.lower() for term in 
                                         ['pergunta', 'questão', 'query', 'question', 'prompt', 'q', 'input'])), None)
                            a_field = next((k for k in item if any(term in k.lower() for term in 
                                         ['resposta', 'answer', 'response', 'reply', 'a', 'output'])), None)
                            
                            if q_field and a_field:
                                qa_pairs.append(f"Pergunta: {item[q_field]}\nResposta: {item[a_field]}")
                            else:
                                # Sem campos específicos, usar o texto completo
                                qa_pairs.append(json.dumps(item, ensure_ascii=False))
                elif isinstance(data, dict):
                    # Verificar estruturas comuns
                    if "conversations" in data and isinstance(data["conversations"], list):
                        for conv in data["conversations"]:
                            if isinstance(conv, dict) and "q" in conv and "a" in conv:
                                qa_pairs.append(f"Pergunta: {conv['q']}\nResposta: {conv['a']}")
                    elif "questions" in data and "answers" in data:
                        questions = data["questions"]
                        answers = data["answers"]
                        if len(questions) == len(answers):
                            for q, a in zip(questions, answers):
                                qa_pairs.append(f"Pergunta: {q}\nResposta: {a}")
                    else:
                        # Usar apenas o objeto principal
                        qa_pairs.append(json.dumps(data, ensure_ascii=False))
        
        logger.info(f"Carregados {len(qa_pairs)} exemplos de {filepath}")
        return qa_pairs
    except Exception as e:
        logger.error(f"Erro ao processar arquivo JSON {filepath}: {str(e)}")
        return []

def load_pdf_file(filepath):
    """Carrega texto de um arquivo PDF"""
    if not PDF_SUPPORT:
        logger.warning("PyPDF2 não está instalado. Instale com 'pip install PyPDF2'")
        return []
    
    text_content = []
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
        
        combined_text = " ".join(text_content)
        
        # Dividir em parágrafos
        paragraphs = re.split(r'\n\s*\n', combined_text)
        
        # Processar parágrafos para identificar pares de pergunta/resposta
        qa_pairs = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Detectar padrões de pergunta/resposta
            qa_match = re.search(r'(Pergunta|P|Q)[\s:.]+(.+?)\s+(Resposta|R|A)[\s:.]+(.+)', paragraph, re.IGNORECASE | re.DOTALL)
            if qa_match:
                q = qa_match.group(2).strip()
                a = qa_match.group(4).strip()
                qa_pairs.append(f"Pergunta: {q}\nResposta: {a}")
            elif paragraph.strip().endswith('?') and i + 1 < len(paragraphs):
                # Possível pergunta seguida de resposta
                qa_pairs.append(f"Pergunta: {paragraph}\nResposta: {paragraphs[i+1].strip()}")
            else:
                # Tratar como texto geral
                qa_pairs.append(paragraph)
        
        logger.info(f"Carregados {len(qa_pairs)} exemplos de {filepath}")
        return qa_pairs
    except Exception as e:
        logger.error(f"Erro ao processar arquivo PDF {filepath}: {str(e)}")
        return []

def load_docx_file(filepath):
    """Carrega texto de um arquivo DOCX"""
    if not DOCX_SUPPORT:
        logger.warning("python-docx não está instalado. Instale com 'pip install python-docx'")
        return []
    
    try:
        doc = docx.Document(filepath)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        
        # Processar parágrafos para identificar pares de pergunta/resposta
        qa_pairs = []
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.startswith("Pergunta:") or paragraph.endswith("?"):
                if i + 1 < len(paragraphs) and (paragraphs[i+1].startswith("Resposta:") or not paragraphs[i+1].endswith("?")):
                    q = paragraph.replace("Pergunta:", "").strip()
                    a = paragraphs[i+1].replace("Resposta:", "").strip()
                    qa_pairs.append(f"Pergunta: {q}\nResposta: {a}")
                else:
                    qa_pairs.append(paragraph)
            elif not (i > 0 and (paragraphs[i-1].startswith("Pergunta:") or paragraphs[i-1].endswith("?"))):
                qa_pairs.append(paragraph)
        
        logger.info(f"Carregados {len(qa_pairs)} exemplos de {filepath}")
        return qa_pairs
    except Exception as e:
        logger.error(f"Erro ao processar arquivo DOCX {filepath}: {str(e)}")
        return []