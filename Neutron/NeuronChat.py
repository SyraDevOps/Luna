import sqlite3
import torch
import torch.nn as nn
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Cerebro:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_path = f'{self.model_name}.pt'
        self.contexto_atual = None
        
        # Carregar o modelo
        self.modelo = None
        self.vectorizer = TfidfVectorizer()
        self.label_to_index = {}
        self.index_to_label = {}
        self.carregar_modelo()

    def carregar_modelo(self):
        """Carrega o modelo principal"""
        if os.path.exists(self.model_path):
            with torch.serialization.safe_globals([SimpleNN]):
                self.modelo = torch.load(self.model_path, weights_only=False)
            with open(f'{self.model_name}_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(f'{self.model_name}_label_mapping.pkl', 'rb') as f:
                self.label_to_index, self.index_to_label = pickle.load(f)
            print(f"Modelo '{self.model_name}' carregado!")
        else:
            print(f"Modelo '{self.model_name}' não encontrado!")

    def processar_pergunta(self, pergunta):
        """Processa a pergunta do usuário considerando contexto."""
        # Concatena contexto (se houver) com a pergunta
        pergunta_com_contexto = f"{self.contexto_atual} {pergunta}" if self.contexto_atual else pergunta
        
        if self.modelo:
            try:
                X = self.vectorizer.transform([pergunta_com_contexto]).toarray()
                X = torch.tensor(X, dtype=torch.float32)
                self.modelo.eval()
                with torch.no_grad():
                    outputs = self.modelo(X)
                    _, predicted = torch.max(outputs, 1)
                    resposta = self.index_to_label[predicted.item()]
                    return resposta
            except Exception as e:
                return f"Erro: {str(e)}"
        return "Não sei responder isso ainda..."

def listar_modelos():
    """Lista todos os modelos disponíveis (arquivos .pt)"""
    modelos = set()
    for file in os.listdir():
        if file.endswith('.pt'):
            modelos.add(file[:-3])  # Remove a extensão .pt
    return modelos

def main():
    print("=== Chat com Modelo de IA ===")
    
    modelos = listar_modelos()
    if modelos:
        print("Modelos disponíveis:")
        for modelo in modelos:
            print(f" - {modelo}")
    else:
        print("Nenhum modelo encontrado.")
        return
    
    modelo = input("Nome do modelo: ").strip()
    if modelo not in modelos:
        print(f"Modelo '{modelo}' não encontrado.")
        return
    
    cerebro = Cerebro(modelo)
    
    while True:
        entrada = input("\nVocê: ").strip()
        if entrada.lower() == '/sair':
            break
        
        resposta = cerebro.processar_pergunta(entrada)
        print(f"Sistema: {resposta}")
    
    print("Chat encerrado!")
