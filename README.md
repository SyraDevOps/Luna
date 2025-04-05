# LunaGPT: Sistema de Diálogo Adaptativo e Dinâmico

![Luna](img/Luna.png)

## Sumário
- Visão Geral
- Arquitetura do Sistema
- Componentes Arquiteturais Avançados
- Instalação
- Guia de Uso
- Fluxo de Trabalho
- Preparação de Dados
- Otimização de Performance
- Funcionalidades Avançadas
- Testes Unitários e Qualidade
- Resolução de Problemas
- Contribuição
- Licença

## Visão Geral

LunaGPT é um sistema de diálogo avançado projetado para interações dinâmicas em português, com foco em adaptabilidade contextual e personalização. O sistema integra técnicas de ponta em processamento de linguagem natural com uma arquitetura modular e expansível.

### Principais Diferenciais

- **Arquitetura Híbrida**: Combinação de transformers com camadas state-space e mixture of experts
- **Adaptabilidade Dinâmica**: Ajuste em tempo real a diferentes perfis de hardware e requisitos de memória
- **Contextualização Proativa**: Sistema de sugestões baseado em padrões de diálogo
- **Curriculum Learning**: Treinamento em estágios progressivos para estabilidade e eficiência
- **RAG Integrado**: Enriquecimento contextual via recuperação de documentos relevantes
- **Sistema de Feedback**: Aprendizado contínuo através de avaliações de interações

## Arquitetura do Sistema

### Estrutura de Diretórios

```
LunaGPT/
├── data/                  # Datasets de treinamento e validação
│   ├── train/            # Arquivos de texto para treinamento
│   └── valid/            # Arquivos de texto para validação
├── logs/                  # Logs do sistema
├── models/                # Checkpoints de modelos salvos
├── src/                   # Código-fonte
│   ├── chat/             # Componentes de interface de chat
│   │   ├── luna_chat.py           # Interface principal de chat
│   │   └── proactive_messenger.py # Sistema de sugestões proativas
│   ├── config/           # Sistema de configuração
│   │   └── config.py              # Classes de configuração
│   ├── models/           # Arquitetura de modelo
│   │   ├── feedback_system.py     # Sistema de feedback
│   │   ├── growing_network.py     # Rede neural expansível
│   │   ├── hypernet.py            # Hiperrede para geração de parâmetros
│   │   ├── luna_model.py          # Modelo principal
│   │   ├── moe.py                 # Mixture of Experts
│   │   ├── rag_retriever.py       # Sistema RAG
│   │   └── tokenizer.py           # Tokenizador personalizado
│   ├── tests/            # Testes unitários e integração
│   ├── training/         # Componentes de treinamento
│   │   └── trainer.py             # Sistema de treinamento
│   └── utils/            # Funções utilitárias
├── feedback.jsonl         # Feedback coletado de usuários
├── main.py                # Interface CLI principal
└── setup.py               # Script de instalação
```

### Core Components

- **LunaModel**: Rede neural principal com componentes avançados (MoE, State-Space Layer, HyperNetwork)
- **LunaTokenizer**: Tokenizador customizado para texto em português com tratamento de tokens especiais
- **LunaTrainer**: Sistema de treinamento com suporte a curriculum learning e callbacks personalizados
- **LunaChat**: Interface de diálogo interativo com sistema proativo de sugestões
- **RAGRetriever**: Sistema de recuperação de documentos para enriquecimento contextual
- **FeedbackSystem**: Coleta e aplicação de feedback para melhoria contínua

## Componentes Arquiteturais Avançados

### 1. Arquitetura Híbrida e Modular

#### Mix of Experts (MoE) com Roteamento Emocional
```python
# Exemplo de uso do MoE
moe_block = MoEBlock(
    input_dim=768,
    num_experts=4,
    sparse_top_k=2,
    emotional_routing=True
)
output = moe_block(inputs, emotion_weights)
```

O MoE permite que diferentes "especialistas" (sub-redes neurais) sejam ativados seletivamente dependendo da entrada, aumentando a eficiência computacional. O roteamento emocional incorpora fatores como empatia, curiosidade e formalidade para direcionar quais especialistas são utilizados.

#### HyperNetwork para Geração Dinâmica de Parâmetros

```python
# Exemplo de uso do HyperNetwork
hypernet = HyperNetwork(context_dim=64, target_dim=768)
weight, bias = hypernet(context_vector)
```

A HyperNetwork gera parâmetros para outras camadas condicionados ao contexto, permitindo adaptação rápida a diferentes domínios ou tipos de entrada.

#### GrowingNetwork para Expansão durante Treinamento

```python
# Exemplo de uso do GrowingNetwork
growing_net = GrowingNetwork(base_model)
# Após algum ponto do treinamento
if growing_net.should_grow(current_loss):
    growing_net.add_layer(input_dim=256, output_dim=256)
```

Esta inovação permite que a rede neural cresça durante o treinamento, adicionando novas camadas conforme necessário para capturar mais capacidade quando o modelo atingir certos marcos.

#### State-Space Layer para Dependências de Longo Alcance

```python
# Exemplo de uso do StateSpaceLayer
ssl = StateSpaceLayer(hidden_size=768, state_size=64)
output = ssl(input_sequence)
```

Camada inspirada em modelos state-space para modelagem eficiente de dependências de longo alcance em sequências, complementando o mecanismo de atenção dos transformers.

### 2. Pipeline de Treinamento Avançado

#### Curriculum Learning Multistágio

```python
# Exemplo de curriculum learning
stages = [
    {"context_length": 128, "batch_size": 16},
    {"context_length": 256, "batch_size": 8},
    {"context_length": 512, "batch_size": 4},
]
for stage in stages:
    trainer.train_supervised(train_data, valid_data, **stage)
```

Treinamento em estágios progressivos, começando com sequências curtas e lotes maiores, e aumentando gradualmente a complexidade.

#### Callbacks Personalizados

- **CheckpointPTCallback**: Salva pesos em múltiplos formatos
- **EarlyStoppingCallback**: Interrompe o treinamento quando a métrica de validação não melhora
- **DynamicHyperparamCallback**: Ajusta o learning rate em tempo real
- **MemoryReplayCallback**: Re-treina com exemplos de feedback de alta qualidade

### 3. Sistema RAG (Retrieval-Augmented Generation)

```python
# Exemplo de uso do RAG
retriever = RAGRetriever(embedding_dim=768)
retriever.add_documents(documents, metadatas)
relevant_docs = retriever.retrieve("Como funciona o LunaGPT?", top_k=3)
```

O sistema RAG recupera documentos relevantes para enriquecer o contexto do modelo durante a geração de resposta, integrando conhecimento externo.

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- GPU compatível com CUDA (recomendado para treinamento)
- 8GB+ RAM (16GB+ recomendado)

### Instruções de Setup

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/luna-project.git
   cd luna-project
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv Luna
   # No Windows
   Luna\Scripts\activate
   # No Linux/macOS
   source Luna/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -e .
   
   # Para instalar dependências para GPU
   pip install -e ".[gpu]"
   
   # Para instalar dependências de desenvolvimento
   pip install -e ".[dev]"
   ```

## Guia de Uso

### Interface de Linha de Comando

LunaGPT oferece uma interface de comando com várias operações:

### Criação de Modelo

```bash
python main.py create --name MODEL_NAME [--train_data PATH_PATTERN [PATH_PATTERN ...]]
```

**Exemplos**:

```bash
# Criar modelo com texto de amostra padrão
python main.py create --name LunaGPT-v1

# Criar modelo com dados personalizados
python main.py create --name LunaGPT-v1 --train_data "data/train/*.txt" "extra_data/*.json" "docs/*.pdf"
```

### Treinamento

```bash
python main.py train --model MODEL_NAME [--epochs NUM_EPOCHS] [--train_data PATH_PATTERN [PATH_PATTERN ...]] [--valid_data PATH_PATTERN [PATH_PATTERN ...]] [--use_wandb] [--lightweight]
```

**Exemplos**:

```bash
# Treinamento básico
python main.py train --model LunaGPT-v1

# Treinamento com opções personalizadas
python main.py train --model LunaGPT-v1 --epochs 5 --train_data "custom_data/*.txt" --use_wandb

# Treinamento em máquina com recursos limitados
python main.py train --model LunaGPT-v1 --lightweight
```

### Chat Interativo

```bash
python main.py chat --model MODEL_NAME [--context INITIAL_CONTEXT] [--persona {tecnico,casual,formal}] [--device {auto,cpu,cuda,mps}]
```

**Exemplos**:

```bash
# Chat básico
python main.py chat --model LunaGPT-v1

# Chat com persona formal e contexto inicial
python main.py chat --model LunaGPT-v1 --persona formal --context "Bom dia, preciso de ajuda com um projeto."

# Chat forçando CPU (útil quando há problemas de memória GPU)
python main.py chat --model LunaGPT-v1 --device cpu
```

### Refinamento com Feedback

```bash
python main.py refine --model MODEL_NAME [--lightweight] [--device {auto,cpu,cuda,mps}]
```

**Exemplos**:

```bash
# Refinamento básico
python main.py refine --model LunaGPT-v1

# Refinamento em hardware limitado
python main.py refine --model LunaGPT-v1 --lightweight --device cpu
```

### Execução de Testes

```bash
python main.py test [--minimal]
```

## Fluxo de Trabalho

### Fluxo Completo de Treinamento

```bash
# 1. Criar diretórios necessários
mkdir -p data/train data/valid

# 2. Adicionar dados de treinamento
echo "Este é um exemplo de diálogo para treinar o modelo." > data/train/sample.txt
# Adicionar mais arquivos conforme necessário

# 3. Criar novo modelo
python main.py create --name MeuLunaGPT

# 4. Treinar modelo (com integração Weights & Biases)
python main.py train --model MeuLunaGPT --epochs 3 --use_wandb

# 5. Interagir com o modelo treinado
python main.py chat --model MeuLunaGPT --persona casual

# 6. Após coletar feedback, refinar o modelo
python main.py refine --model MeuLunaGPT
```

### Fluxo Avançado com RAG

```bash
# 1. Criar e treinar o modelo base
python main.py create --name LunaRAG
python main.py train --model LunaRAG --epochs 3

# 2. Criar script para população do RAG
cat > populate_rag.py << 'EOL'
from src.models.rag_retriever import RAGRetriever
import glob

# Criar retriever
retriever = RAGRetriever(embedding_dim=768)

# Carregar documentos
documents = []
for file in glob.glob("knowledge_base/*.txt"):
    with open(file, 'r', encoding='utf-8') as f:
        documents.append(f.read())

# Adicionar documentos ao retriever
retriever.add_documents(documents)

# Salvar retriever
retriever.save("models/LunaRAG/retriever")
print(f"RAG populado com {len(documents)} documentos")
EOL

# 3. Executar script de população
python populate_rag.py

# 4. Interagir com o modelo enriquecido por RAG
python main.py chat --model LunaRAG
```

## Preparação de Dados

### Formatos Suportados

O sistema suporta múltiplos formatos de entrada:
- Arquivos de texto (.txt)
- CSV (.csv)
- JSON/JSONL (.json, .jsonl)
- Documentos PDF (.pdf)
- Microsoft Word (.docx)

### Exemplo de Dados de Treinamento

```
Usuário: Olá, como posso usar o LunaGPT?
Luna: Olá! Você pode usar o LunaGPT através da interface de linha de comando. Basta executar "python main.py chat --model [nome-do-modelo]" para iniciar uma conversa.

Usuário: Quais são os recursos avançados do LunaGPT?
Luna: O LunaGPT possui vários recursos avançados, como Mixture of Experts, State-Space Layers, HyperNetwork e RAG para enriquecimento contextual. Essas tecnologias permitem respostas mais precisas e adaptáveis.
```

### Data Augmentation

O sistema pode realizar augmentação automática de dados, incluindo:
- Reordenação sintática
- Substituição por sinônimos
- Adição de variações de estilo

## Otimização de Performance

### Configurações para Hardware Limitado

```bash
# Treinamento em hardware limitado
python main.py train --model MeuLunaGPT --lightweight --device cpu

# Chat em hardware limitado
python main.py chat --model MeuLunaGPT --lightweight --device cpu
```

### Quantização e Compressão

O LunaGPT implementa automaticamente técnicas de quantização dinâmica e poda (pruning) para reduzir o tamanho do modelo e acelerar a inferência:

```python
# A quantização é aplicada automaticamente em hardware limitado
# Para forçar quantização em hardware potente:
config.model.force_quantization = True

# Para ajustar o nível de quantização:
config.model.quantization_bits = 8  # 8, 4 ou 2 bits
```

### Monitoramento com Weights & Biases

```bash
# Treinar com integração WandB
python main.py train --model MeuLunaGPT --use_wandb
```

## Funcionalidades Avançadas

### Sistema de Personas

LunaGPT suporta três personas principais que afetam o estilo de resposta:

- **tecnico**: Respostas detalhadas, precisas e objetivas
- **casual**: Tom conversacional, amigável e informal
- **formal**: Linguagem profissional, estruturada e educada

```bash
# Alternar entre personas
python main.py chat --model MeuLunaGPT --persona tecnico
python main.py chat --model MeuLunaGPT --persona casual
python main.py chat --model MeuLunaGPT --persona formal
```

### Proatividade Contextual

O sistema detecta padrões de conversação como:

- Indecisão: "não sei", "talvez", "estou em dúvida"
- Solicitações de ajuda: "como funciona", "preciso de ajuda"
- Busca de informações: "onde posso encontrar", "procurar por"

E oferece sugestões proativas para auxiliar o usuário:

```
Usuário: Não sei qual modelo de LunaGPT eu deveria usar para o meu projeto.

[Sugestão]: Posso explicar mais detalhes para te ajudar a decidir?
```

### Curriculum Learning

```bash
# Script para treinamento com curriculum
cat > train_curriculum.py << 'EOL'
from src.config.config import Config
from src.training.trainer import LunaTrainer

config = Config()
trainer = LunaTrainer("MeuLunaGPT", config)

# Definir estágios do curriculum
stages = [
    {"context_length": 64, "batch_size": 32, "epochs": 1},
    {"context_length": 128, "batch_size": 16, "epochs": 2},
    {"context_length": 256, "batch_size": 8, "epochs": 2}
]

# Carregar dados
train_data = [...] # seus dados de treinamento
valid_data = [...] # seus dados de validação

# Treinar com curriculum
for i, stage in enumerate(stages):
    print(f"Iniciando estágio {i+1}/{len(stages)}")
    config.training.per_device_train_batch_size = stage["batch_size"]
    config.training.max_position_embeddings = stage["context_length"]
    config.training.num_train_epochs = stage["epochs"]
    trainer.train_supervised(train_data, valid_data)
EOL
```

## Testes Unitários e Qualidade

### Executando Testes

```bash
# Executar todos os testes
python main.py test

# Executar testes com configurações mínimas (para hardware limitado)
python main.py test --minimal

# Executar um módulo de teste específico
python -m unittest src.tests.test_chat
```

### Cobertura de Testes

O sistema inclui testes abrangentes para todos os componentes:

- **Arquitetura**: MoEBlock, HyperNetwork, GrowingNetwork, StateSpaceLayer
- **Tokenização**: Treinamento, configuração de tokens especiais
- **Modelo**: Criação, salvamento, carregamento
- **Chat**: Resposta, extração, personas
- **RAG**: Indexação, recuperação
- **Pipeline**: Treinamento completo, feedback, refinamento
- **Proatividade**: Detecção de padrões, sugestões contextuais

## Resolução de Problemas

### Erros Comuns de Criação de Modelo

**Erro**: `'LunaModel' object has no attribute 'save_pretrained'`

**Solução**: A API foi atualizada para usar `save()` em vez de `save_pretrained()`. Verifique se está usando a versão mais recente.

### Erros de Treinamento

**Erro**: `'TrainingConfig' object has no attribute 'num_train_epochs'`

**Solução**: Verifique a classe TrainingConfig em config.py para garantir que tenha o atributo `num_train_epochs`.

### Problemas de Memória

**Erro**: "CUDA out of memory"

**Solução**:
- Use a flag `--lightweight` para configurações otimizadas
- Force CPU com `--device cpu`
- Reduza `batch_size` e aumente `gradient_accumulation_steps`

### Problemas de Dependências

**Error**: "No module named 'faiss'"

**Solução**:
```bash
pip install faiss-cpu
# OU para suporte GPU
pip install faiss-gpu
```

## Contribuição

1. Fork o repositório e crie um branch para sua feature
2. Siga o estilo de código existente
3. Adicione testes para novas funcionalidades
4. Atualize a documentação conforme necessário
5. Envie um pull request com uma descrição clara das mudanças

### Padrão de Commit

```
[COMPONENT] Brief description

Detailed explanation of changes
```

Exemplo:
```
[MoE] Add emotional routing capability

- Implement emotion embedding layer
- Add support for emotion-conditioned routing
- Update tests for the new functionality
```

## Licença

Este projeto está licenciado sob MIT License - consulte o arquivo LICENSE para detalhes.

---

Desenvolvido com 💙 pelo time Syra.

Para questões ou suporte, abra uma issue no repositório do projeto.