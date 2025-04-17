import os
import json
import argparse
from tqdm import tqdm
import logging
from datasets import load_dataset
import requests
from huggingface_hub import hf_hub_download
import random

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Criar diretórios
def setup_directories():
    """Cria os diretórios necessários para os datasets."""
    dirs = ["data", "data/train", "data/valid", "data/test", "data/combined"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Diretório verificado/criado: {d}")

# Função para salvar dataset em formato JSONL
def save_dataset(data, filepath, format_func=None):
    """Salva o dataset em formato JSONL."""
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in tqdm(data, desc=f"Salvando {os.path.basename(filepath)}"):
            if format_func:
                processed_entry = format_func(entry)
                if processed_entry:  # Só salva se retornar algo válido
                    json.dump({"text": processed_entry}, f, ensure_ascii=False)
                    f.write("\n")
            else:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
    logger.info(f"Dataset salvo em: {filepath}")
    return filepath

# Funções de formatação específicas para cada dataset
def format_redial(entry):
    """Formata entradas do dataset ReDial."""
    if "response" in entry and "context" in entry:
        return f"Contexto: {entry['context']}\nResposta: {entry['response']}"
    return None

def format_coraa(entry):
    """Formata entradas do dataset CORAA."""
    if "text" in entry:
        return entry["text"]
    return None

def format_alpaca(entry):
    """Formata entradas do dataset Alpaca-ptbr no formato instrução."""
    if all(k in entry for k in ["instruction", "output"]):
        text = f"Instrução: {entry['instruction']}\n"
        if entry.get('input') and entry['input'].strip():
            text += f"Entrada: {entry['input']}\n"
        text += f"Resposta: {entry['output']}"
        return text
    return None

def format_parliament(entry):
    """Formata entradas do dataset Parliament."""
    if "text" in entry:
        return entry["text"]
    return None

def format_news(entry):
    """Formata entradas do dataset BrNews."""
    if "title" in entry and "text" in entry:
        return f"Título: {entry['title']}\n\n{entry['text']}"
    return None

def format_faqad(entry):
    """Formata entradas do dataset FaQAD-GP."""
    if "question" in entry and "answer" in entry:
        return f"Pergunta: {entry['question']}\nResposta: {entry['answer']}"
    return None

def format_instruction(entry):
    """Formata entradas de datasets genéricos de instrução."""
    if "instruction" in entry and "output" in entry:
        text = f"Instrução: {entry['instruction']}\n"
        if "input" in entry and entry["input"]:
            text += f"Entrada: {entry['input']}\n"
        text += f"Resposta: {entry['output']}"
        return text
    return None

# Funções para cada dataset específico
def download_redial_ptbr(base_dir="data"):
    """Baixa e processa o dataset ReDial em português."""
    logger.info("Baixando dataset ReDial-PTBR...")
    try:
        dataset = load_dataset("matheusrdgsf/re_dial_ptbr")
        train_data = dataset["train"]
        
        # Dividir em treino (90%) e validação (10%)
        train_size = int(0.9 * len(train_data))
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]
        
        train_examples = [train_data[i] for i in train_indices]
        valid_examples = [train_data[i] for i in valid_indices]
        
        # Salvar
        train_path = os.path.join(base_dir, "train", "re_dial_ptbr_train.jsonl")
        valid_path = os.path.join(base_dir, "valid", "re_dial_ptbr_valid.jsonl")
        
        save_dataset(train_examples, train_path, format_redial)
        save_dataset(valid_examples, valid_path, format_redial)
        
        return train_path, valid_path
    except Exception as e:
        logger.error(f"Erro ao baixar ReDial-PTBR: {str(e)}")
        return None, None

def download_alpaca_ptbr(base_dir="data"):
    """Baixa e processa o dataset Alpaca em português."""
    logger.info("Baixando dataset Alpaca-PTBR...")
    try:
        dataset = load_dataset("victorlcampos/alpaca-ptbr-dataset")
        train_data = dataset["train"]
        
        # Dividir em treino (95%) e validação (5%)
        train_size = int(0.95 * len(train_data))
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]
        
        train_examples = [train_data[i] for i in train_indices]
        valid_examples = [train_data[i] for i in valid_indices]
        
        # Salvar
        train_path = os.path.join(base_dir, "train", "alpaca_ptbr_train.jsonl")
        valid_path = os.path.join(base_dir, "valid", "alpaca_ptbr_valid.jsonl")
        
        save_dataset(train_examples, train_path, format_alpaca)
        save_dataset(valid_examples, valid_path, format_alpaca)
        
        return train_path, valid_path
    except Exception as e:
        logger.error(f"Erro ao baixar Alpaca-PTBR: {str(e)}")
        return None, None

def download_portuguese_instructions(base_dir="data"):
    """Baixa e processa o dataset de instruções em português."""
    logger.info("Baixando dataset Open-Instruction-Generalist-Portuguese...")
    try:
        dataset = load_dataset("treelogai/open-instruction-generalist-portuguese")
        train_data = dataset["train"]
        
        # Salvar
        train_path = os.path.join(base_dir, "train", "instructions_ptbr_train.jsonl")
        save_dataset(train_data, train_path, format_instruction)
        
        return train_path, None
    except Exception as e:
        logger.error(f"Erro ao baixar Portuguese Instructions: {str(e)}")
        return None, None

def download_coraa(base_dir="data"):
    """Baixa e processa o dataset CORAA de conversas transcritas."""
    logger.info("Baixando dataset CORAA...")
    try:
        dataset = load_dataset("gobbli/coraa")
        train_data = dataset["train"]
        
        # Salvar
        train_path = os.path.join(base_dir, "train", "coraa_train.jsonl")
        save_dataset(train_data, train_path, format_coraa)
        
        return train_path, None
    except Exception as e:
        logger.error(f"Erro ao baixar CORAA: {str(e)}")
        return None, None

def download_portuguese_parliament(base_dir="data"):
    """Baixa e processa o dataset de debates parlamentares."""
    logger.info("Baixando dataset Portuguese Parlamento...")
    try:
        dataset = load_dataset("portuguese-parlamento", split="train")
        
        # Limitar a 10k amostras para evitar datasets enormes
        if len(dataset) > 10000:
            dataset = dataset.select(range(10000))
        
        # Salvar
        train_path = os.path.join(base_dir, "train", "parliament_ptbr_train.jsonl")
        save_dataset(dataset, train_path, format_parliament)
        
        return train_path, None
    except Exception as e:
        logger.error(f"Erro ao baixar Portuguese Parliament: {str(e)}")
        return None, None

def create_combined_dataset(saved_datasets, base_dir="data", ratio=None):
    """Cria um dataset combinado a partir dos datasets salvos."""
    if not saved_datasets:
        logger.warning("Nenhum dataset para combinar")
        return None
    
    logger.info("Criando dataset combinado...")
    all_data = []
    
    # Se ratio não for especificado, usar proporções iguais
    if not ratio:
        ratio = {os.path.basename(path).split('_')[0]: 1 for path, _ in saved_datasets if path}
    
    # Carregar dados de cada dataset
    for path, _ in saved_datasets:
        if not path:
            continue
            
        dataset_name = os.path.basename(path).split('_')[0]
        dataset_ratio = ratio.get(dataset_name, 1)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            
        # Aplicar proporção (amostragem)
        if dataset_ratio < 1:
            sample_size = int(len(data) * dataset_ratio)
            data = random.sample(data, sample_size)
        
        all_data.extend(data)
    
    # Embaralhar dados
    random.shuffle(all_data)
    
    # Salvar dataset combinado
    combined_path = os.path.join(base_dir, "combined", "combined_ptbr_train.jsonl")
    
    with open(combined_path, "w", encoding="utf-8") as f:
        for entry in tqdm(all_data, desc=f"Salvando dataset combinado"):
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    
    logger.info(f"Dataset combinado salvo em: {combined_path}")
    logger.info(f"Total de exemplos no dataset combinado: {len(all_data)}")
    
    return combined_path

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Baixa e processa datasets em português brasileiro para o modelo Luna.")
    
    parser.add_argument("--datasets", nargs="+", choices=["all", "redial", "alpaca", "instructions", "coraa", "parliament"],
                      default=["all"], help="Datasets a serem baixados")
    
    parser.add_argument("--combine", action="store_true", help="Combinar todos os datasets baixados")
    
    parser.add_argument("--output_dir", type=str, default="data", 
                      help="Diretório para salvar os datasets")
    
    args = parser.parse_args()
    
    # Configurar diretórios
    setup_directories()
    
    # Lista para acompanhar datasets salvos
    saved_datasets = []
    
    # Baixar datasets selecionados
    datasets_to_download = ["redial", "alpaca", "instructions", "coraa", "parliament"] if "all" in args.datasets else args.datasets
    
    if "redial" in datasets_to_download:
        train_path, valid_path = download_redial_ptbr(args.output_dir)
        if train_path:
            saved_datasets.append((train_path, "conversational"))
    
    if "alpaca" in datasets_to_download:
        train_path, valid_path = download_alpaca_ptbr(args.output_dir)
        if train_path:
            saved_datasets.append((train_path, "instruction"))
    
    if "instructions" in datasets_to_download:
        train_path, _ = download_portuguese_instructions(args.output_dir)
        if train_path:
            saved_datasets.append((train_path, "instruction"))
    
    if "coraa" in datasets_to_download:
        train_path, _ = download_coraa(args.output_dir)
        if train_path:
            saved_datasets.append((train_path, "conversational"))
    
    if "parliament" in datasets_to_download:
        train_path, _ = download_portuguese_parliament(args.output_dir)
        if train_path:
            saved_datasets.append((train_path, "conversational"))
    
    # Combinar datasets se solicitado
    if args.combine and saved_datasets:
        combined_path = create_combined_dataset(saved_datasets, args.output_dir)
        if combined_path:
            logger.info(f"Dataset combinado criado com sucesso em {combined_path}")
    
    logger.info("Processamento de datasets concluído!")

if __name__ == "__main__":
    main()