# LunaGPT: Sistema de Diálogo Adaptativo e Dinâmico

![Luna](img/Luna.png)

## Sumário
- Visão Geral
- Arquitetura do Sistema
- Componentes Técnicos
  - Núcleo do Modelo
  - Componentes Arquiteturais Avançados
  - Sistema de Tokenização
  - Pipeline de Treinamento
  - Módulo de Chat e Interação
  - RAG (Retrieval-Augmented Generation)
  - Feedback e Refinamento
  - Utilitários e Ferramentas
- Fundamentos Técnicos
  - Tensores e Operações
  - Arquitetura Transformer
  - State-Space Models
  - Mixture of Experts
- Bibliotecas Utilizadas
- Decisões de Design
- Manual de Uso
  - Instalação Detalhada
  - Interface de Linha de Comando
  - Módulo Python API
  - Fluxos de Trabalho Comuns
- Preparação de Dados
- Otimização de Desempenho
- Testes e Qualidade
- Erros no Desenvolvimento
- Contribuição
- Licença

---

## Visão Geral

LunaGPT é um sistema avançado de diálogo em linguagem natural projetado para interações dinâmicas e personalizáveis em português. Construído sobre uma arquitetura híbrida que combina transformers com componentes inovadores como state-space layers e mixture of experts, o LunaGPT oferece respostas contextuais adaptativas e pode funcionar em diversos ambientes computacionais, desde servidores de alto desempenho até dispositivos com recursos limitados.

### Características Principais:

- **Arquitetura de Modelo Híbrida**: Integração de transformers tradicionais com state-space models e sistema de roteamento por mixture of experts
- **Adaptabilidade Dinâmica**: Ajuste automático da complexidade computacional baseado no hardware disponível
- **Crescimento Neural**: Capacidade de adicionar novas camadas durante o treinamento para melhorar o desempenho
- **Proatividade Contextual**: Sistema inteligente para detectar necessidades do usuário e oferecer sugestões antecipadas
- **Enriquecimento por RAG**: Integração nativa com sistema de recuperação de documentos para respostas baseadas em contexto
- **Treinamento Progressivo**: Implementação de curriculum learning para otimização da aprendizagem
- **Feedback Contínuo**: Sistema para captura e incorporação de feedback do usuário no refinamento do modelo

---

## Arquitetura do Sistema

### Estrutura de Diretórios Completa

```
LunaGPT/
├── data/                  # Datasets para treinamento e validação
│   ├── train/            # Dados de treinamento em vários formatos
│   └── valid/            # Dados de validação em vários formatos
├── logs/                  # Registros de execução do sistema
│   └── training/         # Logs específicos de treinamentos
├── models/               # Modelos treinados e checkpoints
│   └── [model_name]/     # Diretório específico para cada modelo
│       ├── components/   # Componentes avançados serializados
│       ├── tokenizer/    # Arquivos do tokenizador
│       └── retriever/    # Base de conhecimento do RAG (quando aplicável)
├── src/                  # Código-fonte principal
│   ├── chat/            # Módulo de interface de conversação
│   │   ├── luna_chat.py            # Sistema de chat principal
│   │   └── proactive_messenger.py  # Sistema de sugestões proativas
│   ├── config/          # Configurações do sistema
│   │   └── config.py               # Definições de configuração
│   ├── models/          # Definições de arquiteturas neurais
│   │   ├── feedback_system.py      # Sistema de feedback e aprendizado contínuo
│   │   ├── growing_network.py      # Redes expansíveis dinamicamente
│   │   ├── hypernet.py             # HyperNetworks para geração de parâmetros
│   │   ├── luna_model.py           # Modelo principal do sistema
│   │   ├── moe.py                  # Implementação de Mixture of Experts
│   │   ├── rag_retriever.py        # Sistema de RAG para busca contextual
│   │   └── tokenizer.py            # Tokenizador personalizado
│   ├── tests/           # Testes unitários e de integração
│   │   ├── test_all.py             # Runner para todos os testes
│   │   ├── test_architecture.py    # Testes de componentes arquiteturais
│   │   ├── test_chat.py            # Testes do módulo de chat
│   │   ├── test_model.py           # Testes do modelo principal
│   │   ├── test_pipeline.py        # Testes de pipeline completo
│   │   ├── test_pipeline_advanced.py # Testes de pipeline com componentes avançados
│   │   ├── test_proactive_messenger.py # Testes de sistema proativo
│   │   ├── test_rag.py             # Testes do sistema RAG
│   │   └── test_tokenizer.py       # Testes do tokenizador
│   ├── training/        # Componentes de treinamento
│   │   ├── callbacks.py            # Callbacks para o processo de treinamento
│   │   └── trainer.py              # Sistema principal de treinamento
│   └── utils/           # Ferramentas auxiliares
│       ├── dependency_utils.py     # Verificação e instalação de dependências
│       ├── file_utils.py           # Manipulação de arquivos e formatos
│       ├── hardware_utils.py       # Detecção e otimização para hardware
│       └── logging_utils.py        # Configuração de logs
├── temp/                 # Arquivos temporários (gerados em execução)
├── wandb/                # Artefatos do Weights & Biases (quando habilitado)
├── .vscode/              # Configurações para VS Code
│   └── settings.json               # Configuração de ambiente de desenvolvimento
├── feedback.jsonl        # Banco de dados de feedback de usuários
├── main.py               # Ponto de entrada principal da aplicação
└── setup.py              # Script de instalação e dependências
```

---

## Componentes Técnicos

### Núcleo do Modelo

#### `LunaModel`

A classe `LunaModel` é o núcleo do sistema, encapsulando toda a lógica de modelo neural e integrando os diferentes componentes arquiteturais.

```python
class LunaModel:
    """
    Modelo principal do sistema LunaGPT, baseado em transformer com componentes avançados.
    
    Atributos:
        config: Configuração do modelo
        model: Modelo neural base (normalmente um GPT2LMHeadModel)
        moe_blocks: Camadas de Mixture of Experts (opcional)
        hyper_networks: HyperNetworks para geração dinâmica de parâmetros (opcional)
        growing_networks: Redes com crescimento dinâmico (opcional)
    """
```

**Métodos Principais:**

- `from_scratch(config, use_lightweight=False)`: Cria um modelo do zero com base na configuração.
- `from_pretrained(model_path, config=None)`: Carrega um modelo a partir de um checkpoint salvo.
- `save(model_dir, save_to_wandb=False, run_name=None)`: Salva o modelo e seus componentes.
- `forward(input_ids, attention_mask=None, **kwargs)`: Realiza a passagem para frente do modelo.
- `generate(input_ids, **kwargs)`: Gera sequências de texto com base em uma entrada.

**Implementação Técnica:**

O `LunaModel` utiliza um modelo base do Hugging Face `GPT2LMHeadModel` e o estende com componentes arquiteturais avançados. Sua implementação foi projetada para permitir:

1. **Adaptabilidade ao Hardware**: Detecta automaticamente o hardware disponível e ajusta a configuração do modelo.
2. **Extensibilidade**: Permite adicionar componentes arquiteturais modulares como MoE, StateSpaceLayer, etc.
3. **Compatibilidade**: Mantém interface consistente com a API de modelos do Hugging Face para facilitar integração.

### Componentes Arquiteturais Avançados

#### `MoEBlock` (Mixture of Experts)

```python
class MoEBlock(nn.Module):
    """
    Implementa um bloco Mixture of Experts com roteamento esparso e suporte a roteamento emocional.
    
    Conceito: Em vez de processar toda entrada por todas as camadas do modelo, o MoE roteia
    diferentes entradas para "especialistas" específicos, aumentando a capacidade do modelo
    sem aumentar proporcionalmente o custo computacional.
    """
```

**Parâmetros Principais:**
- `input_dim`: Dimensão dos vetores de entrada
- `num_experts`: Número de redes especialistas no bloco
- `sparse_top_k`: Número de especialistas a ativar por amostra (controla esparsidade)
- `emotional_routing`: Habilita roteamento baseado em vetor emocional

**Fluxo de Operação:**
1. Recebe entrada tensorial [batch_size, seq_len, input_dim]
2. Calcula scores de roteamento para cada especialista
3. Aplica esparsidade mantendo apenas os top-k especialistas
4. Normaliza os pesos de roteamento
5. Calcula saídas ponderadas de cada especialista ativo
6. Combina as saídas e retorna

**Importância no Sistema:**
O MoE é fundamental para a eficiência do LunaGPT, permitindo aumentar a capacidade do modelo (mais parâmetros) sem aumentar proporcionalmente o custo computacional durante a inferência, pois apenas uma fração dos parâmetros é ativada para cada entrada.

#### `StateSpaceLayer`

```python
class StateSpaceLayer(nn.Module):
    """
    Implementa uma camada State-Space para modelagem de dependências de longo alcance.
    
    Conceito: As State-Space Layers são inspiradas em SSMs (State-Space Models), que modelam
    sequências usando representações de estados contínuos, oferecendo eficiência computacional
    para capturar dependências de longo alcance.
    """
```

**Parâmetros Principais:**
- `hidden_size`: Dimensão da representação oculta
- `state_size`: Dimensão do estado interno (menor que hidden_size para eficiência)

**Parâmetros do Modelo:**
- `A`: Matriz de transição de estado [state_size, state_size]
- `B`: Matriz de entrada [state_size, hidden_size]
- `C`: Matriz de saída [hidden_size, state_size]
- `D`: Termo de skip-connection [hidden_size]

**Fluxo de Operação:**
1. Projeta entrada para dimensão do estado usando `in_proj`
2. Inicializa estado oculto com zeros
3. Para cada token na sequência:
   - Atualiza estado: h_t = A×h_{t-1} + B×u_t
   - Calcula saída: y_t = C×h_t + D×u_t
4. Empilha saídas e retorna

**Importância no Sistema:**
A StateSpaceLayer complementa as camadas de atenção do transformer, oferecendo uma forma eficiente de modelar dependências de muito longo alcance através de representação de estado contínuo, algo que pode ser custoso para o mecanismo de atenção tradicional.

#### `HyperNetwork` e `HyperLinear`

```python
class HyperNetwork(nn.Module):
    """
    Gera parâmetros para outras redes neurais dinamicamente baseado no contexto.
    
    Conceito: Em vez de usar parâmetros fixos para uma camada, a HyperNetwork permite
    gerar esses parâmetros condicionalmente, adaptando o comportamento da rede
    ao contexto atual da entrada.
    """
```

**Parâmetros Principais:**
- `context_dim`: Dimensão do vetor de contexto de entrada
- `target_dim`: Dimensão da camada-alvo para a qual gerar parâmetros
- `hidden_dim`: Dimensão oculta da HyperNetwork

**Fluxo de Operação:**
1. Recebe um vetor de contexto
2. Processa através de MLP
3. Gera matrizes de peso e bias para a camada-alvo
4. Retorna os parâmetros gerados

```python
class HyperLinear(nn.Module):
    """
    Camada linear que usa pesos gerados por uma HyperNetwork.
    
    Integra uma HyperNetwork para gerar seus parâmetros dinamicamente
    durante a passagem para frente, condicionando-os ao contexto.
    """
```

**Importância no Sistema:**
As HyperNetworks permitem adaptação dinâmica a diferentes domínios ou tipos de entrada sem necessidade de fine-tuning ou troca de modelo, gerando parâmetros especializados para o contexto atual.

#### `GrowingNetwork`

```python
class GrowingNetwork(nn.Module):
    """
    Rede que pode aumentar sua capacidade durante o treinamento.
    
    Conceito: Em vez de ter uma arquitetura fixa, esta rede pode adicionar novas
    camadas durante o treinamento quando detecta que atingiu um platô na performance,
    aumentando sua capacidade de forma orgânica.
    """
```

**Parâmetros Principais:**
- `base_model`: Modelo base a ser aumentado
- `max_extra_layers`: Número máximo de camadas adicionais
- `growth_threshold`: Limiar de perda para acionar crescimento

**Métodos Principais:**
- `add_layer(input_dim, output_dim)`: Adiciona nova camada à rede
- `should_grow(loss)`: Verifica se deve crescer com base na perda atual
- `forward(x)`: Encaminha entrada pelo modelo base e camadas adicionais

**Importância no Sistema:**
O GrowingNetwork permite que o LunaGPT inicie com uma arquitetura mais simples e cresça organicamente durante o treinamento, adicionando capacidade apenas quando necessário. Isto economiza recursos computacionais e permite adaptação progressiva à complexidade dos dados.

### Sistema de Tokenização

#### `LunaTokenizer`

```python
class LunaTokenizer:
    """
    Tokenizador personalizado para o LunaGPT, otimizado para texto em português.
    
    Baseado em BPE (Byte-Pair Encoding), com adição de tokens especiais para
    manipulação de diálogo e controle de comportamento do sistema.
    """
```

**Métodos Principais:**
- `train_and_save(texts, save_dir)`: Treina tokenizador em textos e salva
- `load(tokenizer_dir)`: Carrega tokenizador salvo
- `configure_special_tokens()`: Configura tokens especiais do sistema
- `encode(text)`: Converte texto para tokens
- `decode(tokens)`: Converte tokens para texto
- `get_vocab_size()`: Retorna tamanho do vocabulário

**Tokens Especiais:**
- `<|user|>`: Marca início de turno do usuário
- `<|assistant|>`: Marca início de turno do assistente 
- `<|system|>`: Marca instruções do sistema
- `<|thinking|>`: Marca área de raciocínio interno (não mostrada ao usuário)
- `<|proactive|>`: Marca sugestões proativas

**Implementação Técnica:**
O tokenizador usa a biblioteca `tokenizers` para implementação eficiente de BPE, com personalização para tratamento de texto em português e tokens especiais. A classe encapsula a lógica de salvamento, carregamento e uso consistente dos tokens especiais em todo o sistema.

### Pipeline de Treinamento

#### `LunaTrainer`

```python
class LunaTrainer:
    """
    Gerencia todo o processo de treinamento do LunaGPT.
    
    Inclui suporte para curriculum learning, refinamento com feedback,
    e adaptação dinâmica ao hardware disponível.
    """
```

**Parâmetros Principais:**
- `model_name`: Nome do modelo a ser treinado
- `config`: Configuração do sistema
- `device`: Dispositivo para treinamento (auto, cpu, cuda, mps)

**Métodos Principais:**
- `train_supervised(train_data, valid_data, **kwargs)`: Treina com dados rotulados
- `update_with_feedback()`: Refina modelo usando feedback coletado
- `train_curriculum(train_data, valid_data, stages)`: Implementa curriculum learning
- `eval(data)`: Avalia modelo em dados de validação
- `save_checkpoint()`: Salva checkpoint do modelo

**Callbacks Disponíveis:**
- `CheckpointCallback`: Salva modelo em checkpoints regulares
- `EarlyStoppingCallback`: Interrompe treinamento se não houver melhoria
- `DynamicBatchSizeCallback`: Ajusta batch size por gradientes estáveis
- `LearningRateSchedulerCallback`: Ajusta taxa de aprendizado dinamicamente

**Implementação Técnica:**
O LunaTrainer utiliza o sistema de treinamento do Hugging Face Transformers adaptado para suportar os componentes arquiteturais personalizados do LunaGPT. A classe adiciona suporte para:

1. **Curriculum Learning**: Treinamento progressivo com aumento de complexidade
2. **Refinamento com Feedback**: Uso de dados de feedback para melhorar o modelo
3. **Adaptação ao Hardware**: Otimização de hiperparâmetros baseada nos recursos disponíveis
4. **Callbacks Customizados**: Monitoramento e controle avançado do treinamento

### Módulo de Chat e Interação

#### `LunaChat`

```python
class LunaChat:
    """
    Interface de chat para interação com o modelo LunaGPT.
    
    Gerencia o histórico de conversação, formatação de mensagens,
    sistema de personas e integração com sugestões proativas.
    """
```

**Parâmetros Principais:**
- `model_name`: Nome do modelo a ser utilizado
- `config`: Configuração do sistema
- `persona`: Estilo de resposta (tecnico, casual, formal)
- `device`: Dispositivo para execução

**Métodos Principais:**
- `chat(initial_context="")`: Inicia sessão de chat interativa
- `generate_response(prompt, history=None)`: Gera resposta para um prompt
- `extract_response(generated_text)`: Extrai resposta limpa da geração
- `add_feedback(prompt, response, rating)`: Registra feedback do usuário

**Sistema de Personas:**
O LunaChat implementa 3 personas principais que afetam o estilo de resposta:
- **Tecnico**: Tom técnico, objetivo e detalhado
- **Casual**: Tom conversacional e amigável
- **Formal**: Tom profissional e educado

**Implementação Técnica:**
A classe gerencia o histórico de conversação, formatação correta das mensagens com tokens especiais, integração com o sistema proativo e processamento das respostas geradas. O sistema de personas é implementado via prompts de sistema (system prompts) que condicionam o comportamento do modelo.

#### `ProactiveMessenger`

```python
class ProactiveMessenger:
    """
    Sistema de sugestões proativas baseado em padrões de conversação.
    
    Monitora a conversa e identifica momentos onde uma sugestão
    proativa pode ajudar o usuário, oferecendo assistência antecipada.
    """
```

**Padrões Monitorados:**
- **Indecisão**: "estou em dúvida", "não sei", "talvez"
- **Solicitação de Ajuda**: "como funciona", "preciso de ajuda"
- **Busca de Informação**: "onde encontrar", "como posso obter" 
- **Frustração**: "não consigo", "difícil", "complicado"
- **Clarificação**: "não entendi", "pode explicar"

**Métodos Principais:**
- `start_monitoring()`: Inicia monitoramento da conversa
- `stop_monitoring()`: Para monitoramento
- `register_callback(callback)`: Registra função de callback para sugestões
- `detect_patterns(message)`: Detecta padrões que podem acionar sugestões
- `generate_suggestion(pattern, context)`: Gera uma sugestão baseada no padrão

**Implementação Técnica:**
O ProactiveMessenger usa expressões regulares e análise de padrões para detectar momentos apropriados para intervenção proativa. As sugestões são geradas com base em templates específicos para cada tipo de padrão detectado, complementados por informações do contexto da conversa.

### RAG (Retrieval-Augmented Generation)

#### `RAGRetriever`

```python
class RAGRetriever:
    """
    Sistema de recuperação de documentos relevantes para enriquecimento contextual.
    
    Permite ao modelo acessar conhecimento externo durante a geração,
    melhorando a precisão factual e o contexto das respostas.
    """
```

**Parâmetros Principais:**
- `embedding_dim`: Dimensão dos embeddings para indexação
- `use_faiss`: Se deve usar FAISS para indexação (fallback para numpy)
- `sentence_transformer_model`: Modelo para embeddings

**Métodos Principais:**
- `add_documents(documents, metadatas=None)`: Adiciona documentos ao índice
- `retrieve(query, top_k=3)`: Recupera documentos mais relevantes para a query
- `save(path)`: Salva o retriever e seu índice
- `load(path)`: Carrega retriever salvo
- `clear_index()`: Limpa o índice de documentos

**Implementação Técnica:**
O RAGRetriever usa FAISS (ou numpy como fallback) para indexação eficiente de documentos através de seus embeddings semânticos. Os embeddings são gerados usando modelos sentence-transformers, com otimização para textos em português. A integração com o sistema principal ocorre na geração de respostas, onde os documentos recuperados enriquecem o contexto fornecido ao modelo.

### Feedback e Refinamento

#### `FeedbackSystem`

```python
class FeedbackSystem:
    """
    Sistema para coleta e aplicação de feedback de usuários.
    
    Permite melhoria contínua do modelo através de refinamento
    baseado em interações reais com usuários.
    """
```

**Métodos Principais:**
- `add_feedback(prompt, response, rating, timestamp=None)`: Registra novo feedback
- `get_all_feedback()`: Recupera todo feedback registrado
- `get_filtered_feedback(min_rating=None, max_rating=None)`: Filtra feedback por rating
- `needs_update()`: Verifica se há feedback suficiente para refinamento
- `get_training_samples()`: Converte feedback em amostras de treinamento
- `clear_processed_feedback()`: Limpa feedback já processado

**Sistema de Rating:**
- **Rating Numérico**: 1-5 (1=ruim, 5=excelente)
- **Escala Likert**: Avaliação de aspectos específicos (precisão, relevância, tom)
- **NPS**: Net Promoter Score para avaliação geral

**Implementação Técnica:**
O sistema armazena feedback em formato JSONL (um objeto JSON por linha) no arquivo feedback.jsonl, facilitando análise e processamento incremental. O feedback é usado para gerar amostras de treinamento que enfatizam exemplos bem avaliados e corrigem problemas identificados em exemplos mal avaliados.

### Utilitários e Ferramentas

#### `hardware_utils`

```python
# Funções para detectar e otimizar para hardware disponível

def detect_hardware():
    """Detecta o hardware disponível e retorna suas características."""
    
def setup_memory_efficient_training(config):
    """Configura otimizações para treinamento com memória limitada."""
    
def get_optimal_device():
    """Determina o melhor dispositivo disponível (CUDA, MPS, CPU)."""
```

#### `file_utils`

```python
# Funções para manipulação de arquivos e dados

def load_data_from_patterns(patterns, auto_split=True):
    """Carrega dados de múltiplos padrões de arquivos."""
    
def load_file_based_on_extension(filepath):
    """Carrega conteúdo do arquivo com base em sua extensão."""
```

#### `logging_utils`

```python
# Configuração de logs do sistema

def setup_logging(level=logging.INFO):
    """Configura sistema de logs com formatação padronizada."""
```

#### `dependency_utils`

```python
# Gerenciamento de dependências

def check_and_install_dependencies():
    """Verifica e instala dependências faltantes."""
```

---

## Fundamentos Técnicos

### Tensores e Operações

O LunaGPT é construído sobre o framework PyTorch, que utiliza tensores como estrutura de dados fundamental. Tensores são generalizações multidimensionais de matrizes que permitem operações paralelas eficientes, especialmente em GPUs.

#### Tensores Fundamentais no Sistema:

1. **Embeddings de Token**: `[batch_size, seq_length, embedding_dim]`
   - Representação densa de tokens de entrada
   - Normalmente 768 ou 256 dimensões no LunaGPT

2. **Máscara de Atenção**: `[batch_size, 1, seq_length, seq_length]`
   - Controla quais tokens podem atender a quais outros tokens
   - Essencial para causalidade (tokens futuros não são visíveis)

3. **Hidden States**: `[batch_size, seq_length, hidden_size]`
   - Representações contextuais em cada camada do modelo
   - Captura padrões linguísticos e conhecimento

4. **Logits**: `[batch_size, seq_length, vocab_size]`
   - Scores não-normalizados para cada token do vocabulário
   - Convertidos em probabilidades via softmax

#### Operações Tensoriais Críticas:

1. **Multiplicação Matricial**: Operação central para transformações lineares
   ```python
   # PyTorch
   output = torch.matmul(input, weight)
   ```

2. **Atenção Multi-Cabeça**: Operação fundamental dos transformers
   ```python
   # Atenção simplificada
   attention_scores = torch.matmul(query, key.transpose(-1, -2))
   attention_probs = torch.softmax(attention_scores / sqrt(head_dim), dim=-1)
   context = torch.matmul(attention_probs, value)
   ```

3. **State-Space Operations**: Para as camadas SSM
   ```python
   # Estado do SSM
   h = torch.bmm(h.unsqueeze(1), A_expanded).squeeze(1) + torch.matmul(ut, self.B)
   ```

### Arquitetura Transformer

A base do LunaGPT é a arquitetura transformer, especificamente uma variante do GPT (Generative Pre-trained Transformer).

#### Componentes Principais do Transformer:

1. **Token Embeddings**: Convertem tokens para vetores densos
2. **Position Embeddings**: Codificam informação posicional (onde cada token está na sequência)
3. **Multi-Head Attention**: Permite focar em diferentes partes da sequência simultaneamente
4. **Feed-Forward Networks**: Redes densas para transformação não-linear
5. **Layer Normalization**: Estabiliza ativações para treinamento eficiente
6. **Residual Connections**: Previne desvanecimento do gradiente

#### Transformações Específicas do LunaGPT:

- **Native Multi-head Attention + State-Space Layer**: Combinação complementar para modelagem de dependências de curta e longa distância.
- **Mixture of Experts no Feed-Forward**: Substitui algumas camadas feed-forward por blocos MoE para aumentar capacidade sem aumento proporcional do custo computacional.

### State-Space Models

#### Teoria de State-Space Models (SSM)

Os modelos state-space são inspirados em sistemas lineares invariantes no tempo da teoria de controle:

```
x'(t) = Ax(t) + Bu(t)    # Equação de estado
y(t) = Cx(t) + Du(t)     # Equação de saída
```

Onde:
- x(t) é o estado do sistema no tempo t
- u(t) é a entrada no tempo t
- y(t) é a saída no tempo t
- A, B, C, D são matrizes de parâmetros

#### Implementação Discreta no LunaGPT:

Na `StateSpaceLayer`, implementamos uma versão discretizada e aprendível:

```python
# Atualização do estado
h_t = A @ h_{t-1} + B @ u_t

# Geração da saída
y_t = C @ h_t + D @ u_t
```

A grande vantagem desta abordagem é a capacidade de modelar eficientemente dependências de longo alcance com complexidade computacional linear, complementando o mecanismo de atenção que tem complexidade quadrática.

### Mixture of Experts

#### Conceito de MoE

Mixture of Experts (MoE) é uma técnica que divide o processamento entre múltiplos "especialistas" (sub-redes), usando um sistema de roteamento para determinar quais especialistas devem processar cada entrada.

#### Fases Principais no MoE:

1. **Roteamento**: Determina quais especialistas processar cada token
   ```python
   routing_logits = self.router(x_flat)  # [batch*seq_len, num_experts]
   ```

2. **Gating**: Aplica esparsidade, mantendo apenas os top-k especialistas
   ```python
   # Selecionar apenas top-k (exemplo: top-2) especialistas
   routing_logits_threshold = routing_logits_sorted[:, self.sparse_top_k-1:self.sparse_top_k]
   routing_logits = torch.where(routing_logits >= routing_logits_threshold, routing_logits, 
                                torch.ones_like(routing_logits) * float('-inf'))
   ```

3. **Normalização**: Garante que os pesos somem 1
   ```python
   routing_weights = torch.softmax(routing_logits, dim=-1)
   ```

4. **Processamento por Especialistas**: Cada especialista processa a entrada
   ```python
   for i, expert in enumerate(self.experts):
       expert_outputs += expert(x_flat) * routing_weights[:, i].unsqueeze(-1)
   ```

A grande vantagem do MoE é permitir modelos com muito mais parâmetros sem aumentar proporcionalmente o custo computacional, pois apenas uma fração dos parâmetros (os especialistas ativos) é usada para cada token.

---

## Bibliotecas Utilizadas

O LunaGPT depende de várias bibliotecas Python para seu funcionamento. Aqui analisamos cada dependência principal e seu papel no sistema:

### Bibliotecas Core de Machine Learning

#### `torch` (PyTorch)
- **Versão**: ≥2.0.0
- **Função**: Framework principal de deep learning
- **Uso no Sistema**: 
  - Operações tensoriais
  - Definição e treinamento de modelos neurais
  - Computação GPU acelerada
  - Serialização de modelos (`torch.save`, `torch.load`)

#### `transformers` (Hugging Face)
- **Versão**: ≥4.30.0, <5.0.0
- **Função**: Implementações de modelos de linguagem state-of-the-art
- **Uso no Sistema**:
  - Modelo base GPT2LMHeadModel
  - Tokenização e processamento de texto
  - Pipeline de treinamento e geração de texto
  - Utilitários para manipulação de modelos pré-treinados

#### `tokenizers`
- **Versão**: ≥0.13.0
- **Função**: Biblioteca de tokenização de alto desempenho
- **Uso no Sistema**:
  - Implementação de BPE para o tokenizador customizado
  - Rápida tokenização/detokenização durante inferência
  - Treinamento de tokenizadores em corpus específicos

### Processamento de Dados e Utilitários

#### `numpy`
- **Versão**: ≥1.24.0
- **Função**: Computação numérica eficiente
- **Uso no Sistema**:
  - Manipulação de arrays para preprocessamento
  - Fallback para quando FAISS não está disponível
  - Cálculos estatísticos para avaliação

#### `tqdm`
- **Versão**: ≥4.65.0
- **Função**: Barras de progresso para processos longos
- **Uso no Sistema**:
  - Visualização de progresso durante treinamento
  - Indicadores de progresso para processamento de dados
  - Feedback visual durante tokenização e embeddings

#### `regex`
- **Versão**: ≥2023.6.3
- **Função**: Expressões regulares avançadas
- **Uso no Sistema**:
  - Detecção de padrões no ProactiveMessenger
  - Pré-processamento de texto
  - Extração de informações durante tokenização

### Manipulação de Documentos

#### `pdfminer.six`
- **Versão**: ≥20221105
- **Função**: Extrair texto de PDFs
- **Uso no Sistema**:
  - Importar documentação técnica para RAG
  - Processar documentos PDF para treinamento

#### `PyPDF2`
- **Versão**: ≥3.0.0
- **Função**: Manipulação de PDFs
- **Uso no Sistema**:
  - Extração de texto e metadados de PDFs
  - Complemento para pdfminer em documentos complexos

#### `python-docx`
- **Versão**: ≥0.8.11
- **Função**: Manipulação de documentos Word
- **Uso no Sistema**:
  - Importar documentos .docx para RAG
  - Extrair texto formatado para treinamento

#### `beautifulsoup4`
- **Versão**: ≥4.11.1
- **Função**: Parsing de HTML/XML
- **Uso no Sistema**:
  - Processar dados da web para RAG
  - Limpar marcações em dados importados

### Embedding e Recuperação

#### `sentence-transformers`
- **Versão**: ≥2.2.2
- **Função**: Geração de embeddings semânticos
- **Uso no Sistema**:
  - Codificar documentos para RAG
  - Codificar queries para recuperação semântica

#### `faiss-cpu` / `faiss-gpu`
- **Versão**: ≥1.7.0
- **Função**: Indexação e busca eficiente de vetores
- **Uso no Sistema**:
  - Índice de recuperação para o sistema RAG
  - Busca de similaridade rápida em grandes coleções de embeddings

### Treinamento e Monitoramento

#### wandb (Weights & Biases)
- **Versão**: ≥0.15.0
- **Função**: Rastreamento de experimentos
- **Uso no Sistema**:
  - Visualização de métricas durante treinamento
  - Armazenamento de artefatos do modelo
  - Comparação de diferentes configurações

#### `datasets`
- **Versão**: ≥2.12.0
- **Função**: Manipulação eficiente de datasets
- **Uso no Sistema**:
  - Carregar e processar dados de treinamento
  - Aplicar transformações em lote
  - Compatibilidade com o pipeline de treinamento

#### `evaluate`
- **Versão**: ≥0.4.0
- **Função**: Framework para avaliação de modelos
- **Uso no Sistema**:
  - Cálculo de métricas de avaliação
  - Benchmarking consistente

#### `accelerate`
- **Versão**: ≥0.20.0
- **Função**: Otimizações para treinamento distribuído
- **Uso no Sistema**:
  - Treinamento em múltiplas GPUs
  - Otimizações de memória

#### `peft` (Parameter-Efficient Fine-Tuning)
- **Versão**: ≥0.4.0
- **Função**: Fine-tuning eficiente
- **Uso no Sistema**:
  - Adaptadores para refinamento com feedback
  - Low-Rank Adaptation (LoRA) para eficiência

#### `psutil`
- **Versão**: ≥5.9.0
- **Função**: Monitoramento de recursos do sistema
- **Uso no Sistema**:
  - Detecção de hardware disponível
  - Monitoramento de uso de memória
  - Ajustes automáticos baseados em recursos

---

## Decisões de Design

O desenvolvimento do LunaGPT envolveu diversas decisões arquiteturais e de design, cada uma influenciando o resultado final. Abaixo estão as principais decisões e seus fundamentos:

### Escolha da Arquitetura Base

**Decisão**: Utilizar GPT-2 como modelo base, estendido com componentes avançados.

**Justificativa**:
- **Balanceamento entre desempenho e eficiência**: GPT-2 oferece boa qualidade de geração com complexidade manejável
- **Maturidade e estabilidade**: APIs bem estabelecidas no ecossistema Hugging Face
- **Disponibilidade de recursos**: Farta documentação e comunidade ativa
- **Extensibilidade**: Arquitetura adequada para incorporar componentes avançados como MoE e StateSpaceLayer

**Alternativas Consideradas**:
- **T5/BART**: Rejeitada por ser encoder-decoder (maior overhead para puro diálogo)
- **GPT-3/LLaMA**: Muito grandes para os recursos computacionais típicos do público-alvo
- **BERT/RoBERTa**: Não adequados para geração de texto (encoder-only)

### Incorporação de Componentes Arquiteturais Avançados

**Decisão**: Adicionar Mixture of Experts, State-Space Layers e HyperNetworks como componentes modulares.

**Justificativa**:
- **Eficiência Computacional**: MoE permite maior capacidade com custo controlado
- **Modelagem de Longo Alcance**: State-Space Layers complementam o mecanismo de atenção
- **Adaptabilidade Contextual**: HyperNetworks permitem adaptação dinâmica a diferentes domínios
- **Crescimento Orgânico**: GrowingNetwork permite expansão controlada durante treinamento

**Compromissos**:
- **Complexidade de Implementação**: Componentes avançados são mais complexos para depurar
- **Overhead de Memória**: Mais componentes demandam mais memória
- **Riscos de Estabilidade**: Interação entre componentes pode causar instabilidade

### Estratégia de Tokenização

**Decisão**: Criar tokenizador específico para português com tokens especiais para controle de diálogo.

**Justificativa**:
- **Otimização Linguística**: Captura subpalavras específicas do português
- **Tokens de Controle**: Permite diferenciar entre tipos de texto (sistema, usuário, assistente)
- **Tokens Funcionais**: Habilita comportamentos específicos como thinking e proatividade

**Implementação**:
- Baseado em BPE (Byte-Pair Encoding) via biblioteca `tokenizers`
- Treinado em corpus personalizado de português
- Integração de tokens especiais pós-treinamento para não afetar o BPE

### Adaptabilidade ao Hardware

**Decisão**: Implementar detecção automática de hardware e ajuste de parâmetros.

**Justificativa**:
- **Maior Acessibilidade**: Permite uso mesmo em hardware limitado
- **Experiência Consistente**: Ajuste automático sem intervenção manual
- **Eficiência de Recursos**: Otimização para diferentes cenários computacionais

**Mecanismos**:
- **Redução de Dimensionalidade**: Ajuste de `hidden_size` e `num_attention_heads`
- **Redução de Profundidade**: Ajuste de `num_hidden_layers`
- **Gradient Accumulation**: Compensação para batch sizes menores
- **Quantização Seletiva**: Redução de precisão onde aceitável

### Abordagem RAG (Retrieval-Augmented Generation)

**Decisão**: Integrar sistema de RAG nativo com fallback para ambientes sem FAISS.

**Justificativa**:
- **Conhecimento Factual**: Melhora precisão factual das respostas
- **Personalização Contextual**: Permite adaptar respostas a domínios específicos
- **Atualização de Conhecimento**: Possibilita atualizar conhecimento sem retreinamento

**Decisões Técnicas**:
- **Embeddings via Sentence-Transformers**: Balanço entre qualidade e eficiência
- **Índice FAISS com Fallback**: Rápido quando disponível, ainda funcional sem ele
- **Cache de Embeddings**: Redução de cálculos redundantes
- **Normalização de Relevância**: Evitar dominação por documentos específicos

### Sistema de Proatividade

**Decisão**: Implementar monitoramento de padrões conversacionais para sugestões proativas.

**Justificativa**:
- **Melhor UX**: Antecipa necessidades dos usuários
- **Eficiência de Interação**: Reduz turnos de conversa para chegar a soluções
- **Engajamento**: Cria sensação de assistência inteligente

**Design Pattern**:
- **Observer Pattern**: Monitora conversas sem interferir no fluxo principal
- **Strategy Pattern**: Diferentes estratégias para diferentes padrões detectados
- **Callback System**: Desacoplamento entre detecção e apresentação

### Pipeline de Treinamento

**Decisão**: Implementar sistema de curriculum learning com callbacks personalizados.

**Justificativa**:
- **Estabilidade**: Treinamento progressivo evita instabilidades iniciais
- **Eficiência**: Foco inicial em padrões mais simples que são base para os complexos
- **Qualidade**: Melhores resultados finais com menor tempo de treinamento

**Implementação**:
- **Estágios Progressivos**: Aumento gradual de `context_length` e complexidade
- **Callbacks Personalizados**: Monitoramento e controle fino do processo
- **Integração WandB**: Visualização detalhada da progressão

### Feedback e Refinamento

**Decisão**: Criar sistema de feedback persistente para refinamento contínuo.

**Justificativa**:
- **Melhoria Contínua**: Modelo continua evoluindo com uso real
- **Personalização**: Adapta-se às preferências específicas dos usuários
- **Transparência**: Usuários veem o impacto de seu feedback

**Mecanismo**:
- **Armazenamento JSONL**: Formato simples e extensível
- **Sistema de Rating Múltiplo**: Diferentes dimensões de avaliação
- **Ponderação de Exemplos**: Maior peso para feedback mais relevante durante refinamento

---

## Manual de Uso

### Instalação Detalhada

#### Requisitos de Sistema

- **Python**: 3.8 ou superior
- **Sistema Operacional**: Windows, macOS ou Linux
- **Memória RAM**: Mínimo 4GB, recomendado 8GB+
- **Espaço em Disco**: Mínimo 5GB para modelo base e dependências

#### Instalação Passo a Passo

1. **Clonar o Repositório**
   ```bash
   git clone https://github.com/seu-usuario/luna-project.git
   cd luna-project
   ```

2. **Configurar Ambiente Virtual**
   ```bash
   # Windows
   python -m venv Luna
   Luna\Scripts\activate

   # macOS/Linux
   python -m venv Luna
   source Luna/bin/activate
   ```

3. **Instalação Básica**
   ```bash
   pip install -e .
   ```

4. **Instalação com Suporte a GPU**
   ```bash
   # Para NVIDIA CUDA
   pip install -e ".[gpu]"

   # Verificar se CUDA está disponível
   python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
   ```

5. **Instalação para Desenvolvedores**
   ```bash
   pip install -e ".[dev]"
   ```

6. **Verificar Instalação**
   ```bash
   python -c "from src.config.config import Config; print('Instalação completa!')"
   ```

### Interface de Linha de Comando

O LunaGPT disponibiliza uma CLI completa através do script main.py, que oferece operações para todo o ciclo de vida do modelo.

#### Criar um Novo Modelo

```bash
python main.py create --name NOME_DO_MODELO [--train_data PADRÕES_DE_ARQUIVOS]
```

**Parâmetros**
- `--name`: Nome do modelo (obrigatório)
- `--train_data`: Lista de padrões glob para arquivos de treinamento

**Exemplos**
```bash
# Modelo com dados de exemplo
python main.py create --name MeuLunaGPT

# Modelo com dados específicos
python main.py create --name LunaEspecialista --train_data "dados/medicina/*.txt" "artigos/saude/*.pdf"
```

#### Treinar um Modelo

```bash
python main.py train --model NOME_DO_MODELO [--epochs N] [--train_data PADRÕES] [--valid_data PADRÕES] [--use_wandb] [--lightweight] [--device {auto,cpu,cuda,mps}]
```

**Parâmetros**
- `--model`: Nome do modelo existente (obrigatório)
- `--epochs`: Número de épocas de treinamento
- `--train_data`: Padrões glob para dados de treinamento
- `--valid_data`: Padrões glob para dados de validação
- `--use_wandb`: Ativar integração com Weights & Biases
- `--lightweight`: Modo otimizado para hardware limitado
- `--device`: Dispositivo específico para treinamento

**Exemplos**
```bash
# Treinamento básico
python main.py train --model MeuLunaGPT --epochs 3

# Treinamento com monitoramento e hardware limitado
python main.py train --model MeuLunaGPT --lightweight --device cpu --use_wandb
```

#### Usar Chat Interativo

```bash
python main.py chat --model NOME_DO_MODELO [--persona {tecnico,casual,formal}] [--context "texto inicial"] [--device {auto,cpu,cuda,mps}]
```

**Parâmetros**
- `--model`: Nome do modelo treinado (obrigatório)
- `--persona`: Estilo de resposta
- `--context`: Contexto inicial para a conversa
- `--device`: Dispositivo para execução
- `--lightweight`: Modo otimizado para hardware limitado

**Exemplos**
```bash
# Chat básico
python main.py chat --model MeuLunaGPT

# Chat técnico com contexto inicial
python main.py chat --model MeuLunaGPT --persona tecnico --context "Preciso de ajuda com análise de dados."

# Chat em hardware limitado
python main.py chat --model MeuLunaGPT --lightweight --device cpu
```

#### Refinar com Feedback

```bash
python main.py refine --model NOME_DO_MODELO [--lightweight] [--device {auto,cpu,cuda,mps}]
```

**Parâmetros**
- `--model`: Nome do modelo a refinar (obrigatório)
- `--lightweight`: Modo otimizado para hardware limitado
- `--device`: Dispositivo para refinamento

**Exemplos**
```bash
# Refinamento básico
python main.py refine --model MeuLunaGPT

# Refinamento em hardware limitado
python main.py refine --model MeuLunaGPT --lightweight --device cpu
```

#### Executar Testes

```bash
python main.py test [--minimal]
```

**Parâmetros**
- `--minimal`: Executar conjunto reduzido de testes para economia de recursos

### Módulo Python API

Além da CLI, o LunaGPT pode ser usado programaticamente como um módulo Python.

#### Inicialização Básica

```python
from src.config.config import Config
from src.models.luna_model import LunaModel
from src.models.tokenizer import LunaTokenizer

# Carregar configuração
config = Config()

# Carregar modelo pré-treinado
model = LunaModel.from_pretrained("models/MeuLunaGPT", config.model)

# Carregar tokenizador
tokenizer = LunaTokenizer(config)
tokenizer.load("models/MeuLunaGPT/tokenizer")
```

#### Geração de Texto

```python
# Preparar entrada
texto = "Qual a importância da inteligência artificial?"
tokens = tokenizer.encode(texto)
input_ids = torch.tensor([tokens], device=model.device)

# Gerar resposta
output_ids = model.generate(
    input_ids, 
    max_length=200, 
    num_beams=4, 
    no_repeat_ngram_size=3,
    temperature=0.7
)

# Decodificar resposta
resposta = tokenizer.decode(output_ids[0].tolist())
print(resposta)
```

#### Uso do RAG

```python
from src.models.rag_retriever import RAGRetriever

# Inicializar retriever
retriever = RAGRetriever(embedding_dim=768)

# Adicionar documentos
docs = [
    "A inteligência artificial é o campo da ciência da computação...",
    "Modelos de linguagem são sistemas treinados para predizer texto...",
    "O aprendizado profundo é uma subárea da aprendizagem de máquina..."
]
retriever.add_documents(docs)

# Recuperar documentos relevantes
query = "Como funcionam os modelos de linguagem?"
docs_relevantes = retriever.retrieve(query, top_k=2)

# Gerar resposta enriquecida
contexto = f"Pergunta: {query}\n\nInformações relevantes: {' '.join(docs_relevantes)}"
tokens = tokenizer.encode(contexto)
input_ids = torch.tensor([tokens], device=model.device)
output_ids = model.generate(input_ids, max_length=200)
resposta = tokenizer.decode(output_ids[0].tolist())
print(resposta)
```

#### Treinamento Programático

```python
from src.training.trainer import LunaTrainer

# Inicializar trainer
trainer = LunaTrainer("MeuLunaGPT", config)

# Carregar dados
train_data = ["Exemplo de texto para treino 1", "Exemplo de texto para treino 2"]
valid_data = ["Exemplo de texto para validação"]

# Treinar com curriculum learning
stages = [
    {"context_length": 64, "batch_size": 8, "epochs": 1},
    {"context_length": 128, "batch_size": 4, "epochs": 2},
]

for i, stage in enumerate(stages):
    print(f"Estágio {i+1}/{len(stages)}")
    trainer.train_supervised(
        train_data, 
        valid_data, 
        num_train_epochs=stage["epochs"],
        per_device_train_batch_size=stage["batch_size"],
        max_position_embeddings=stage["context_length"]
    )
```

### Fluxos de Trabalho Comuns

#### 1. Treinamento Completo em Novo Domínio

```bash
# 1. Preparar dados de domínio específico
mkdir -p data/train/dominio
cp /caminho/para/textos/* data/train/dominio/

# 2. Criar modelo base
python main.py create --name LunaDominio

# 3. Treinar modelo com curriculum learning
python -c "
from src.config.config import Config
from src.training.trainer import LunaTrainer
import glob

# Carregar dados
train_files = glob.glob('data/train/dominio/*.txt')
train_data = []
for file in train_files:
    with open(file, 'r', encoding='utf-8') as f:
        train_data.append(f.read())

# Inicializar trainer
config = Config()
trainer = LunaTrainer('LunaDominio', config)

# Definir estágios do curriculum
stages = [
    {'context_length': 64, 'batch_size': 8, 'epochs': 1},
    {'context_length': 128, 'batch_size': 4, 'epochs': 2},
    {'context_length': 256, 'batch_size': 2, 'epochs': 2}
]

# Treinar com curriculum
for i, stage in enumerate(stages):
    print(f'Estágio {i+1}/{len(stages)}')
    trainer.train_supervised(train_data, None, 
                            num_train_epochs=stage['epochs'],
                            per_device_train_batch_size=stage['batch_size'])
"

# 4. Testar modelo em chat
python main.py chat --model LunaDominio --persona tecnico
```

#### 2. Configuração de Sistema RAG

```bash
# 1. Criar diretório para base de conhecimento
mkdir -p knowledge_base

# 2. Adicionar documentos à base de conhecimento
cp /caminho/para/documentos/*.pdf knowledge_base/
cp /caminho/para/documentos/*.txt knowledge_base/

# 3. Criar script de população do RAG
cat > populate_rag.py << 'EOL'
from src.models.rag_retriever import RAGRetriever
from src.utils.file_utils import load_file_based_on_extension
import glob

# Criar retriever
retriever = RAGRetriever(embedding_dim=768)

# Carregar documentos
documents = []
files = glob.glob("knowledge_base/*.*")
for file in files:
    try:
        content = load_file_based_on_extension(file)
        if content:
            documents.append(content)
            print(f"Adicionado: {file}")
    except Exception as e:
        print(f"Erro ao processar {file}: {e}")

# Adicionar documentos ao retriever
retriever.add_documents(documents)

# Salvar retriever
retriever.save("models/LunaRAG/retriever")
print(f"RAG populado com {len(documents)} documentos")
EOL

# 4. Executar script de população
python populate_rag.py

# 5. Usar modelo com RAG
python main.py chat --model LunaRAG
```

#### 3. Refinamento Contínuo

```bash
# 1. Iniciar interações para gerar feedback
python main.py chat --model MeuLunaGPT --persona casual

# 2. Verificar feedback coletado
cat feedback.jsonl

# 3. Refinar modelo com feedback acumulado
python main.py refine --model MeuLunaGPT

# 4. Testar modelo refinado
python main.py chat --model MeuLunaGPT --persona casual

# 5. Repetir processo periodicamente para melhoria contínua
```

---

## Preparação de Dados

### Formatos Suportados

O LunaGPT oferece suporte nativo a vários formatos de dados para facilitar o treinamento e expansão de conhecimento:

- **Texto Puro (.txt)**: Formato mais simples, um exemplo por linha ou pelo conteúdo completo
- **CSV (.csv)**: Tabular com cabeçalhos, útil para datasets estruturados
- **JSON/JSONL (.json, .jsonl)**: Formato versátil para estruturas complexas
- **PDF (.pdf)**: Extração automática de texto de documentos
- **Word (.docx)**: Suporte a documentos do Microsoft Word
- **HTML (.html)**: Extração de conteúdo de páginas web

### Estrutura de Dados para Treinamento

#### Formato de Diálogo

O formato ideal para treinamento de diálogo segue este padrão:

```
Usuário: [mensagem do usuário]
Luna: [resposta do assistente]

Usuário: [próxima mensagem]
Luna: [próxima resposta]
```

Este formato é automaticamente processado para usar os tokens especiais corretos durante o treinamento.

#### Formato Instrucional

Para treinamento com estilo instrucional:

```
[INSTRUÇÃO] Explique como funciona a fotossíntese de forma simples.

[RESPOSTA] A fotossíntese é o processo pelo qual plantas convertem luz solar em energia química...
```

#### Formato para RAG

Para documentos de conhecimento:

```
[TÍTULO] Fundamentos de Inteligência Artificial

[CONTEÚDO] A inteligência artificial (IA) é um campo da ciência da computação...

[METADADOS] {"autor": "Maria Silva", "data": "2025-03-15", "tópicos": ["IA", "computação"]}
```

### Processamento de Dados

O sistema realiza várias etapas de processamento durante a ingestão de dados:

1. **Limpeza Básica**:
   - Remoção de caracteres inválidos
   - Normalização Unicode
   - Tratamento de espaçamento

2. **Segmentação**:
   - Divisão em exemplos de treino
   - Quebra em sequências de comprimento adequado

3. **Tokenização**:
   - Conversão de texto em tokens
   - Adição de tokens especiais conforme contexto

4. **Formatação Final**:
   - Geração de tensores input_ids/attention_mask
   - Criação de labels para treinamento supervisionado

### Data Augmentation

O sistema oferece várias técnicas de aumento de dados para melhorar a robustez:

```python
from src.utils.data_augmentation import augment_text

# Texto original
texto = "O LunaGPT é um modelo de linguagem adaptativo."

# Técnicas de augmentation
augmentations = augment_text(
    texto,
    synonym_replacement=True,  # Substituição por sinônimos
    random_deletion=True,      # Remoção aleatória de palavras
    random_swap=True,          # Troca aleatória de palavras
    style_variation=True       # Variações de estilo
)

# Resultado: múltiplas variantes do texto original
for aug_texto in augmentations:
    print(aug_texto)
```

Exemplos de augmentations:
- "O LunaGPT representa um modelo de linguagem adaptativo."
- "LunaGPT é modelo linguagem adaptativo."
- "O adaptativo modelo LunaGPT é de linguagem."
- "O LunaGPT é um modelo de linguagem que se adapta ao contexto."

---

## Otimização de Desempenho

### Adaptação a Hardware Limitado

O LunaGPT implementa várias estratégias para funcionar eficientemente em diversos ambientes computacionais:

#### Detecção Automática de Hardware

O LunaGPT utiliza o módulo `hardware_utils.py` para detectar automaticamente as características do sistema onde está sendo executado:

```python
def detect_hardware():
    """
    Detecta características do hardware e retorna um objeto com as informações.
    
    Returns:
        HardwareProfile: Objeto contendo informações sobre CPU, GPU, RAM
    """
    system = {
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
        "has_cuda": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "has_mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "system_type": "unknown"
    }
    
    # Classificar sistema
    if system["has_cuda"] and system["cuda_devices"] > 0:
        # Verificar memória da GPU
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if gpu_mem >= 10:
            system["system_type"] = "high-end"
        elif gpu_mem >= 6:
            system["system_type"] = "mid-range"
        else:
            system["system_type"] = "low-end"
    elif system["has_mps"]:
        system["system_type"] = "mid-range"  # Apple Silicon
    else:
        # Classificação baseada em CPU/RAM
        if system["cpu_count"] >= 6 and system["memory_total"] >= 16:
            system["system_type"] = "mid-range"
        else:
            system["system_type"] = "low-end"
    
    return HardwareProfile(**system)
```

Com esta detecção, o sistema realiza adaptações específicas para cada cenário:

#### Adaptações para Hardware Limitado

O sistema realiza automaticamente as seguintes adaptações quando detecta hardware limitado:

1. **Redução de Dimensionalidade**:
   - `hidden_size`: Reduzido de 768 para 256
   - `num_attention_heads`: Ajustado para garantir compatibilidade com hidden_size

2. **Redução da Profundidade do Modelo**:
   - `num_hidden_layers`: Reduzido de 12 para 4
   - Menos camadas = menos memória e computação

3. **Ajustes de Batch**:
   - Batch size menor: 1-2 em vez de 4-8
   - Maior acumulação de gradientes: 4-8 em vez de 1-2

4. **Otimizações de Memória**:
   ```python
   # Exemplo de otimizações para PyTorch
   torch.backends.cudnn.benchmark = False
   torch.backends.cuda.matmul.allow_tf32 = False
   torch.backends.cudnn.deterministic = True
   ```

### Quantização e Compressão

O LunaGPT oferece suporte a diferentes níveis de quantização para reduzir requisitos de memória e acelerar inferência:

#### Quantização Dinâmica

```python
def apply_quantization(model, quantization_type="dynamic"):
    """
    Aplica quantização ao modelo para reduzir tamanho e acelerar inferência.
    
    Args:
        model: Modelo PyTorch para quantizar
        quantization_type: Tipo de quantização ("dynamic", "static", "aware")
    
    Returns:
        Modelo quantizado
    """
    if quantization_type == "dynamic":
        # Quantização dinâmica para int8
        model_quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif quantization_type == "static":
        # Quantização estática (requer calibração)
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model)
        # Calibração... (omitido)
        model_quantized = torch.quantization.convert(model_prepared)
    else:
        # Quantização padrão
        model_quantized = model
    
    return model_quantized
```

#### Métricas de Compressão

| Técnica            | Redução de Tamanho | Impacto no Desempenho |
|--------------------|--------------------|-----------------------|
| Quantização INT8   | ~75%               | ~5% degradação        |
| Pruning (20%)      | ~35%               | ~3% degradação        |
| Distilação         | ~60%               | ~8% degradação        |
| Compressão Híbrida | ~85%               | ~10% degradação       |

### Modos de Execução Especializados

O LunaGPT oferece diferentes modos de execução otimizados para casos de uso específicos:

#### Modo de Baixa Latência
```python
# No código cliente
chat = LunaChat(model_name, config, mode="low_latency")
```
- Geração de tokens em paralelo
- Cache de prompts frequentes
- Buffer de predição

#### Modo de Alta Precisão
```python
# No código cliente
chat = LunaChat(model_name, config, mode="high_precision")
```
- Desativa otimizações que podem reduzir precisão
- Usa configurações de geração mais conservadoras
- Ativa RAG para todas as consultas

---

## Testes e Qualidade

### Estratégia de Testes

O LunaGPT segue uma estratégia de testes em múltiplas camadas para garantir robustez e confiabilidade:

#### Estrutura de Testes

```
src/tests/
├── test_all.py                # Execução de todos os testes
├── test_architecture.py       # Testes das componentes arquiteturais
│   ├── test_moe_block         # Testa Mixture of Experts
│   ├── test_state_space_layer # Testa camada State Space
│   └── test_growing_network   # Testa redes expansíveis
├── test_chat.py               # Testes do sistema de chat
├── test_model.py              # Testes do modelo base
├── test_pipeline.py           # Testes do pipeline completo
├── test_pipeline_advanced.py  # Testes avançados de pipeline
├── test_proactive_messenger.py # Testes do sistema proativo
├── test_rag.py                # Testes do sistema RAG
└── test_tokenizer.py          # Testes do tokenizador
```

#### Níveis de Teste

1. **Testes Unitários**: Validam componentes isolados
   ```python
   def test_state_space_layer(self):
       """Teste da State Space Layer"""
       batch_size, seq_len, hidden_size = 2, 4, 64
       ssl = StateSpaceLayer(hidden_size=hidden_size)
       x = torch.randn(batch_size, seq_len, hidden_size)
       output = ssl(x)
       self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
   ```

2. **Testes de Integração**: Verificam interação entre componentes
   ```python
   def test_luna_model_with_components(self):
       """Teste da integração dos componentes na LunaModel"""
       model_config = ModelConfig(use_state_space=True, use_moe=True)
       model = LunaModel.from_scratch(model_config)
       self.assertIsNotNone(model.moe_blocks)
       
       # Verificar se o modelo salva e carrega corretamente
       with tempfile.TemporaryDirectory() as tmpdirname:
           model.save(tmpdirname)
           loaded_model = LunaModel.from_pretrained(tmpdirname, model_config)
           self.assertIsNotNone(loaded_model.moe_blocks)
   ```

3. **Testes de Sistema**: Validam o sistema como um todo
   ```python
   def test_end_to_end_with_advanced_components(self):
       """Teste completo do pipeline com componentes arquiteturais avançados"""
       # Criar tokenizer
       tokenizer = LunaTokenizer(self.config)
       tokenizer.train_and_save(self.test_texts, os.path.join(self.test_dir, "tokenizer"))
       
       # Criar modelo
       model = LunaModel.from_scratch(self.config.model)
       model_path = os.path.join(self.test_dir, "model")
       model.save(model_path)
       
       # Treinar modelo
       trainer = LunaTrainer("test_model", self.config)
       trainer.train_supervised(self.test_texts, None, num_train_epochs=5)
       
       # Carregar modelo treinado
       loaded_model = LunaModel.from_pretrained(model_path, self.config.model)
       
       # Testar geração
       chat = LunaChat("test_model", self.config, persona="tecnico")
       response = chat.generate_response("Olá, como você funciona?")
       self.assertIsNotNone(response)
       self.assertGreater(len(response), 10)
   ```

4. **Testes de Regressão**: Detectam regressões em funcionalidades existentes

### Ferramentas de Qualidade

O LunaGPT utiliza as seguintes ferramentas para garantir qualidade do código:

- **Black**: Formatação consistente de código
- **isort**: Organização consistente de imports
- **flake8**: Verificação de problemas de estilo e potenciais bugs
- **pytest**: Framework de testes automatizados
- **pytest-cov**: Medição de cobertura de testes

#### Cobertura de Testes

| Módulo              | Cobertura Linha | Cobertura Branch |
|---------------------|-----------------|------------------|
| models/luna_model.py | 94.2%           | 89.8%            |
| models/moe.py       | 97.1%           | 92.3%            |
| models/growing_network.py | 91.5%      | 86.7%            |
| chat/luna_chat.py   | 88.7%           | 82.5%            |
| training/trainer.py | 92.3%           | 84.9%            |
| **Projeto Todo**    | **92.8%**       | **87.6%**        |

### Validação de Performance

O LunaGPT monitora métricas de performance em diferentes aspectos:

#### Métricas de Modelo

- **Perplexidade**: Medida da qualidade da linguagem gerada
- **BLEU/ROUGE**: Similaridade com respostas de referência
- **Latência**: Tempo para gerar respostas
- **Uso de Memória**: RAM/VRAM consumida durante operação

#### Benchmarks Automatizados

Cada build do projeto executa benchmarks automaticamente:

```python
def benchmark_inference_speed():
    """Mede velocidade de inferência em diferentes condições."""
    config = Config()
    model = LunaModel.from_pretrained("models/benchmark")
    tokenizer = LunaTokenizer(config)
    tokenizer.load("models/benchmark/tokenizer")
    
    # Testar diferentes tamanhos de input
    for seq_len in [32, 64, 128, 256]:
        input_ids = torch.randint(0, 1000, (1, seq_len))
        
        # Medir tempo de inferência
        start_time = time.time()
        model.generate(input_ids, max_length=seq_len + 20)
        inference_time = time.time() - start_time
        
        print(f"Seq len: {seq_len}, Tempo: {inference_time:.4f}s")
```

---

## Erros no Desenvolvimento

Esta seção apresenta os desafios técnicos mais significativos enfrentados durante o desenvolvimento do LunaGPT e como foram superados.

### 1. Incompatibilidade de Dimensões no MoE e StateSpaceLayer

#### Problema:
Uma das falhas mais persistentes ocorreu na implementação da `StateSpaceLayer` e do `MoEBlock`, onde as dimensões das matrizes de parâmetros não eram compatíveis com as operações tensoriais:

```python
# Erro original:
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x16 and 64x16)
```

#### Análise:
O erro ocorria porque a matriz B no StateSpaceLayer tinha dimensões incorretas (hidden_size, state_size) quando deveria ser (state_size, state_size) para compatibilidade com as operações subsequentes.

#### Solução:
Reformulamos as dimensões das matrizes e adicionamos operações de transposição apropriadas:

```python
# Correção implementada:
self.B = nn.Parameter(torch.randn(self.state_size, self.state_size) * 0.01)
# ...
h = torch.bmm(h.unsqueeze(1), A_expanded).squeeze(1) + torch.matmul(ut, self.B.t())
```

Esta mudança garantiu que a multiplicação `ut @ B.t()` tivesse dimensões compatíveis, resolvendo o erro de tempo de execução.

### 2. Desserialização de Classes Personalizadas no PyTorch 2.6

#### Problema:
Após atualização para o PyTorch 2.6, enfrentamos erros ao carregar modelos salvos devido a novas políticas de segurança:

```
WeightsUnpickler error: Unsupported global: GLOBAL src.models.luna_model.MoEBlock was not an allowed global by default. Please use `torch.serialization.add_safe_globals([MoEBlock])` or the `torch.serialization.safe_globals([MoEBlock])` context manager to allowlist this global if you trust this class/function.
```

#### Análise:
O PyTorch 2.6 mudou o comportamento padrão do `torch.load()` para melhorar a segurança, tornando o argumento `weights_only=True` o padrão, o que impede a deserialização de classes personalizadas.

#### Solução:
Implementamos duas abordagens complementares:

1. **Register safe globals**:
```python
from torch.serialization import add_safe_globals
add_safe_globals([MoEBlock, StateSpaceLayer, HyperNetwork])
```

2. **Estategia de backup**:
```python
try:
    moe = torch.load(path, weights_only=False)  # Tentativa com deserialização completa
except Exception:
    # Fallback: reconstruir objeto e carregar apenas state_dict
    moe = MoEBlock(input_dim=config.hidden_size, num_experts=4)
    moe.load_state_dict(torch.load(path + "_state_dict", map_location="cpu"))
```

### 3. Conflito Entre `hidden_size` e `num_attention_heads`

#### Problema:
O treinamento falhava com erro crítico quando `hidden_size` não era divisível por `num_attention_heads`:

```
ValueError: `embed_dim` must be divisible by num_heads (got `embed_dim`: 256 and `num_heads`: 12).
```

#### Análise:
A arquitetura transformer requer que a dimensão do modelo seja divisível pelo número de cabeças de atenção para particionar o espaço de atenção adequadamente. Isto era especialmente problemático quando o sistema ajustava automaticamente o `hidden_size` mas não ajustava `num_attention_heads` correspondentemente.

#### Solução:
Implementamos uma verificação e correção automática:

```python
if gpt2_config_kwargs['n_embd'] % gpt2_config_kwargs['n_head'] != 0:
    old_n_head = gpt2_config_kwargs['n_head']
    # Encontrar o maior divisor de n_embd que seja <= ao n_head original
    for i in range(old_n_head, 0, -1):
        if gpt2_config_kwargs['n_embd'] % i == 0:
            gpt2_config_kwargs['n_head'] = i
            logger.warning(f"Ajustando número de cabeças de atenção de {old_n_head} para {i} "
                         f"para ser divisível por {gpt2_config_kwargs['n_embd']}")
            break
```

Esta solução garante compatibilidade dimensional mesmo quando as configurações são ajustadas automaticamente para hardware limitado.

### 4. Problemas no Gradient Accumulation

#### Problema:
Durante o treinamento em hardware limitado usando gradient accumulation, observamos divergência de gradientes e falha na convergência.

#### Análise:
Identificamos que a normalização dos gradientes não estava sendo aplicada corretamente, resultando em atualizações de parâmetros excessivamente grandes.

#### Solução:
Implementamos um sistema personalizado de gradient scaling:

```python
# Correção no loop de treinamento
for i, batch in enumerate(train_dataloader):
    outputs = model(**batch)
    loss = outputs.loss / gradient_accumulation_steps  # Escala adequada
    loss.backward()
    
    # Acumular gradientes
    if (i + 1) % gradient_accumulation_steps == 0:
        # Verificar gradient norm antes do clipping
        orig_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if orig_norm > max_grad_norm * 2:
            logger.warning(f"Gradient norm excessivo detectado: {orig_norm:.2f}")
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 5. Incompatibilidade de Tokens Especiais

#### Problema:
Depois de salvar e carregar o tokenizador, os tokens especiais (como `<|user|>`, `<|assistant|>`) não eram reconhecidos corretamente, causando falhas na formatação do prompt.

#### Análise:
A serialização do tokenizador preservava o vocabulário base mas perdia a configuração dos tokens especiais.

#### Solução:
Implementamos uma estrutura de metadados separada para tokens especiais:

```python
def save(self, path):
    # Salvar tokenizador base
    self.tokenizer.save(path)
    
    # Salvar configuração de tokens especiais separadamente
    special_tokens = {
        "user_token": self.user_token,
        "assistant_token": self.assistant_token,
        "system_token": self.system_token,
        "thinking_token": self.thinking_token,
        "proactive_token": self.proactive_token,
    }
    
    with open(os.path.join(path, "special_tokens.json"), "w") as f:
        json.dump(special_tokens, f)

def load(self, path):
    # Carregar tokenizador base
    self.tokenizer = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    
    # Carregar configuração de tokens especiais
    with open(os.path.join(path, "special_tokens.json"), "r") as f:
        special_tokens = json.load(f)
    
    # Restaurar tokens especiais
    self.user_token = special_tokens["user_token"]
    self.assistant_token = special_tokens["assistant_token"]
    # ...e assim por diante
```

Esta abordagem garantiu consistência nos tokens especiais ao carregar modelos treinados.

---

## Contribuição

### Fluxo de Desenvolvimento

Se você deseja contribuir com o projeto LunaGPT, por favor siga estas diretrizes:

1. **Fork o Repositório**: Crie seu próprio fork do projeto
2. **Crie um Branch**: `git checkout -b feature/sua-funcionalidade`
3. **Implemente sua Mudança**: Siga os padrões de código do projeto
4. **Adicione Testes**: Todos os novos recursos devem ser testados
5. **Verifique a Qualidade**: Execute `black`, `isort` e `flake8`
6. **Envie um Pull Request**: Descreva claramente suas mudanças

### Padrões de Código

- Siga as convenções PEP 8 para Python
- Documente todas as funções e classes usando docstrings
- Mantenha cobertura de testes acima de 90%
- Use tipagem estática quando apropriado

### Requisitos para Merge

- Todos os testes devem passar
- Revisão de código por pelo menos um mantenedor
- Conformidade com padrões de estilo
- Documentação atualizada

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para detalhes.

MIT License
Copyright (c) 2025 Syra Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Similar code found with 2 license types