import os
import json
import argparse
import logging
from tqdm import tqdm

def analyze_tokens(model_name, min_frequency=10, max_tokens=100, auto_add=False):
    """Analisa candidatos a tokens e opcionalmente os adiciona ao tokenizer"""
    model_dir = os.path.join("models", model_name)
    token_file = os.path.join(model_dir, "token_candidates.json")
    
    if not os.path.exists(token_file):
        print(f"Arquivo de candidatos a tokens não encontrado: {token_file}")
        return
        
    try:
        # Carregar candidatos
        with open(token_file, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
            
        print(f"Carregados {len(candidates)} candidatos a tokens do arquivo {token_file}")
        
        # Filtrar por frequência
        filtered = {term: count for term, count in candidates.items() 
                   if count >= min_frequency}
        
        # Ordenar por frequência
        sorted_candidates = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        
        # Limitar ao número máximo especificado
        top_candidates = sorted_candidates[:max_tokens]
        
        print(f"\n===== Top {len(top_candidates)} candidatos a tokens =====")
        for i, (term, count) in enumerate(top_candidates, 1):
            print(f"{i}. '{term}' (frequência: {count})")
            
        if not auto_add:
            return
            
        # Adicionar tokens automaticamente se solicitado
        print("\nAdicionando tokens ao tokenizer...")
        
        # Carregar tokenizer e configuração
        from src.config import Config
        from src.models.adaptive_tokenizer import AdaptiveTokenizer
        
        config = Config()
        adaptive_tokenizer = AdaptiveTokenizer(model_name, config)
        
        # Forçar frequência mínima para permitir a adição
        adaptive_tokenizer.min_frequency = 1
        
        # Adicionar termos ao contador
        for term, count in top_candidates:
            adaptive_tokenizer.unknown_terms_counter[term] = count
            
        # Estender tokenizer
        from src.models.luna_model import LunaModel
        model_path = os.path.join(model_dir, "full_model.pt")
        
        if os.path.exists(model_path):
            print("Carregando modelo para atualizar embeddings...")
            import torch.serialization
            from src.models.luna_model import LunaModel
            from src.models.moe import MoEBlock
            from src.models.growing_network import StateSpaceLayer, GrowingNetwork
            from src.models.hypernet import HyperNetwork
            
            torch.serialization.add_safe_globals([LunaModel, MoEBlock, StateSpaceLayer, 
                                                GrowingNetwork, HyperNetwork])
            
            model = torch.load(model_path, weights_only=False)
            num_added = adaptive_tokenizer.extend_tokenizer(model)
            
            if num_added > 0:
                print(f"Salvando modelo atualizado com {num_added} novos tokens...")
                torch.save(model, model_path)
            
        else:
            num_added = adaptive_tokenizer.extend_tokenizer()
            
        print(f"Adicionados {num_added} novos tokens ao vocabulário")
        
    except Exception as e:
        print(f"Erro ao analisar tokens: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisador de tokens candidatos")
    parser.add_argument("--model", required=True, help="Nome do modelo")
    parser.add_argument("--min_freq", type=int, default=10, help="Frequência mínima para considerar um token")
    parser.add_argument("--max_tokens", type=int, default=100, help="Número máximo de tokens a considerar")
    parser.add_argument("--auto_add", action="store_true", help="Adicionar tokens automaticamente")
    
    args = parser.parse_args()
    analyze_tokens(args.model, args.min_freq, args.max_tokens, args.auto_add)