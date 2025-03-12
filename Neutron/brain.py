
import sqlite3
import nltk
import networkx as nx
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nltk.download('punkt')
nltk.download('stopwords')


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

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
        self.db_path = f'Neuron_{self.model_name}.db'
        self.model_path = f'{self.model_name}.pt'
        self.contexto_atual = None
        
        # Inicializa dataset com chaves para o modelo principal e o submodelo
        self.data = {
            "perguntas": [],        # perguntas (concatenadas com contexto, se houver) para o modelo principal
            "perguntas_orig": [],    # perguntas originais, sem contexto (para o submodelo)
            "respostas": [],
            "contextos": []         # contextos (pode ser None)
        }
        
        # Conectar ao banco de dados
        self.conexao = sqlite3.connect(self.db_path)
        self.cursor = self.conexao.cursor()
        self.criar_tabelas()
        
        # Inicializar grafo e carregar dados do BD
        self.grafo = nx.DiGraph()
        self.atualizar_grafo()
        self.carregar_dados()
        
        # Carregar ou criar os modelos
        self.modelo = None
        self.vectorizer = TfidfVectorizer()
        self.label_to_index = {}
        self.index_to_label = {}
        self.carregar_modelo()

    def criar_tabelas(self):
        """Cria as tabelas no banco de dados"""
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS neuronios (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                pergunta TEXT UNIQUE,
                                resposta TEXT,
                                contexto TEXT,
                                vezes INTEGER DEFAULT 1
                             )''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS compromissos (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                descricao TEXT,
                                data DATE,
                                hora TIME,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                             )''')
        self.conexao.commit()

    def atualizar_grafo(self):
        """Atualiza o grafo de conhecimento"""
        self.grafo.clear()
        neuronios = self.buscar_neuronios()
        for pergunta, resposta, contexto in neuronios:
            node_label = f"{contexto}: {pergunta}" if contexto else pergunta
            self.grafo.add_node(node_label)
            self.grafo.add_edge(node_label, resposta)
        print("Grafo de conhecimento atualizado.")

    def buscar_neuronios(self):
        """Retorna todos os neurônios com contexto"""
        self.cursor.execute("SELECT pergunta, resposta, contexto FROM neuronios")
        return self.cursor.fetchall()

    def adicionar_neuronio(self, pergunta, resposta, contexto=None):
        """Adiciona novo neurônio com contexto"""
        try:
            self.cursor.execute('''INSERT INTO neuronios 
                                  (pergunta, resposta, contexto)
                                  VALUES (?, ?, ?)''',
                                  (pergunta, resposta, contexto))
            self.conexao.commit()
            print(f"Aprendi: '{pergunta}' no contexto '{contexto}'")
            self.atualizar_grafo()
        except sqlite3.IntegrityError:
            self.atualizar_neuronio(pergunta, resposta, contexto)
        self.carregar_dados()
        self.treinar_modelo()

    def atualizar_neuronio(self, pergunta, nova_resposta, novo_contexto=None):
        """Atualiza neurônio existente com novo contexto"""
        self.cursor.execute('''SELECT resposta, contexto, vezes 
                            FROM neuronios WHERE pergunta = ?''', (pergunta,))
        resultado = self.cursor.fetchone()
        
        if resultado:
            resposta_atual, contexto_atual, vezes = resultado
            nova_resposta = f"{resposta_atual} / {nova_resposta}"
            if novo_contexto and novo_contexto != contexto_atual:
                novo_contexto = f"{contexto_atual} / {novo_contexto}"
            else:
                novo_contexto = contexto_atual
            self.cursor.execute('''UPDATE neuronios 
                                  SET resposta = ?, contexto = ?, vezes = ? 
                                  WHERE pergunta = ?''',
                                  (nova_resposta, novo_contexto, vezes + 1, pergunta))
            self.conexao.commit()
            print(f"Atualizei: '{pergunta}'")
        else:
            self.adicionar_neuronio(pergunta, nova_resposta, novo_contexto)

    def carregar_dados(self):
        """Carrega dados do banco de dados para os modelos"""
        self.cursor.execute("SELECT pergunta, resposta, contexto FROM neuronios")
        neuronios = self.cursor.fetchall()
        # Reinicia os dados
        self.data["perguntas"] = []
        self.data["perguntas_orig"] = []
        self.data["respostas"] = []
        self.data["contextos"] = []
        for n in neuronios:
            pergunta, resposta, contexto = n
            # Para o modelo principal, se houver contexto, concatenamos com a pergunta
            self.data["perguntas"].append(f"{contexto} {pergunta}" if contexto else pergunta)
            self.data["perguntas_orig"].append(pergunta)
            self.data["respostas"].append(resposta)
            self.data["contextos"].append(contexto)

    def carregar_modelo(self):
        """Carrega ou cria o modelo principal"""
        if os.path.exists(self.model_path):
            self.modelo = torch.load(self.model_path)
            with open(f'{self.model_name}_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(f'{self.model_name}_label_mapping.pkl', 'rb') as f:
                self.label_to_index, self.index_to_label = pickle.load(f)
            print(f"Modelo '{self.model_name}' carregado!")
        else:
            self.modelo = None
            self.carregar_dados()
            if self.data["perguntas"]:
                self.treinar_modelo()
            print(f"Novo modelo '{self.model_name}' criado!")

    def salvar_modelo(self):
        """Salva o modelo principal em disco"""
        torch.save(self.modelo, self.model_path)
        with open(f'{self.model_name}_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(f'{self.model_name}_label_mapping.pkl', 'wb') as f:
            pickle.dump((self.label_to_index, self.index_to_label), f)
        print(f"Modelo salvo: {self.model_path}")

    def treinar_modelo(self, epochs=10):
        """Treina o modelo principal com os dados atuais"""
        if self.data["perguntas"] and self.data["respostas"]:
            X = self.vectorizer.fit_transform(self.data["perguntas"]).toarray()
            y = self.data["respostas"]
            
            # Normalizar os dados
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Mapeia as respostas para índices
            unique_labels = list(set(y))
            self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
            y = [self.label_to_index[label] for label in y]
            
            if len(X) > 1:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                X_train, X_val, y_train, y_val = X, X, y, y
            
            train_dataset = TextDataset(X_train, y_train)
            val_dataset = TextDataset(X_val, y_val)
            
            batch_size = min(32, len(train_dataset))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            input_dim = X_train.shape[1]
            output_dim = len(unique_labels)
            self.modelo = SimpleNN(input_dim, output_dim)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.modelo.parameters(), lr=0.001)
            
            for epoch in range(epochs):
                self.modelo.train()
                for texts, labels in train_loader:
                    texts, labels = torch.tensor(texts, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
                    optimizer.zero_grad()
                    outputs = self.modelo(texts)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                self.modelo.eval()
                val_loss = 0
                with torch.no_grad():
                    for texts, labels in val_loader:
                        texts, labels = torch.tensor(texts, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
                        outputs = self.modelo(texts)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss/len(val_loader)}")
            
            self.salvar_modelo()
        else:
            print("Sem dados para treinar o modelo principal!")

    def treinar_com_csv(self, arquivo, epochs=10):
        """Treina o modelo principal com dados de um arquivo CSV"""
        df = pd.read_csv(arquivo)
        if 'pergunta' in df.columns and 'resposta' in df.columns and 'contexto' in df.columns:
            perguntas = df['pergunta'].tolist()
            respostas = df['resposta'].tolist()
            contextos = df['contexto'].tolist()
            
            self.data["perguntas"] = [f"{contexto} {pergunta}" for contexto, pergunta in zip(contextos, perguntas)]
            self.data["perguntas_orig"] = perguntas
            self.data["respostas"] = respostas
            self.data["contextos"] = contextos
            
            self.treinar_modelo(epochs)
        else:
            print("O arquivo CSV deve conter as colunas 'pergunta', 'resposta' e 'contexto'.")

    def processar_pergunta(self, pergunta):
        """Processa a pergunta do usuário considerando contexto."""
        # Concatena contexto (se houver) com a pergunta
        pergunta_com_contexto = f"{self.contexto_atual} {pergunta}" if self.contexto_atual else pergunta
        
        if self.data["perguntas"]:
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

    def adicionar_compromisso(self, descricao, data, hora):
        """Adiciona novo compromisso"""
        try:
            self.cursor.execute('''INSERT INTO compromissos 
                                (descricao, data, hora)
                                VALUES (?, ?, ?)''',
                                (descricao, data, hora))
            self.conexao.commit()
            print(f"Compromisso adicionado: {data} {hora}")
        except Exception as e:
            print(f"Erro: {str(e)}")

    def listar_compromissos(self):
        """Lista todos os compromissos"""
        self.cursor.execute('''SELECT * FROM compromissos 
                            ORDER BY data, hora''')
        return self.cursor.fetchall()

    def mudar_contexto(self, novo_contexto):
        """Altera o contexto atual manualmente"""
        self.contexto_atual = novo_contexto if novo_contexto != 'none' else None
        print(f"Contexto definido para: {self.contexto_atual}")

    def fechar(self):
        """Fecha a conexão com o banco de dados"""
        self.conexao.close()
