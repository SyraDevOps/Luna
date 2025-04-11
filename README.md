# LunaGPT: Sistema Avançado de Diálogo Neural Adaptativo

!Luna

## Índice Detalhado

1. Visão Geral e Fundamentos
   - O que é o LunaGPT
   - Principais Inovações
   - Diferenciadores Técnicos
   - Arquitetura Híbrida

2. Fundamentos Teóricos
   - Arquitetura Transformer
   - Modelos State-Space
   - Mixture of Experts
   - Redes Neurais Adaptativas
   - Fundamentos de Embeddings e Tokenização

3. Arquitetura Detalhada do Sistema
   - Visão Macro: Componentes Principais
   - Estrutura de Diretórios Completa
   - Fluxo de Dados e Processamento

4. Componentes Técnicos Nucleares
   - LunaModel: O Coração do Sistema
   - Mixture of Experts (MoEBlock)
   - State-Space Layer: Modelagem de Sequência Eficiente
   - HyperNetworks: Geração Dinâmica de Parâmetros
   - GrowingNetwork: Expansão Neural Orgânica
   - Tokenização Especializada para Português

5. Funcionalidades Avançadas
   - Sistema RAG (Retrieval-Augmented Generation)
   - Proatividade Contextual
   - Curriculum Learning e Treinamento Progressivo
   - Sistema de Feedback e Refinamento Contínuo
   - Personas e Adaptabilidade de Resposta

6. Instalação e Configuração
   - Requisitos de Sistema Detalhados
   - Instalação Passo-a-Passo
   - Verificação de Instalação
   - Configuração Avançada
   - Soluções para Problemas Comuns de Instalação

7. Guia de Utilização
   - Primeiros Passos (Para Iniciantes)
   - Interface de Linha de Comando
   - API Python para Integração Programática
   - Fluxos de Trabalho Comuns
   - Cenários de Uso Avançado

8. Gerenciamento de Dados
   - Formatos Suportados e Estruturas
   - Processamento e Preparação de Dados
   - Técnicas de Aumento de Dados
   - Gerenciamento da Base de Conhecimento RAG

9. Treinamento e Otimização
   - Pipeline de Treinamento Completo
   - Implementação Eficiente de Curriculum Learning
   - Treinamento Baseado em Feedback
   - Otimizações para Hardware Variado
   - Quantização e Técnicas de Compressão

10. Testes e Garantia de Qualidade
    - Estratégia de Testes Multinível
    - Métricas de Qualidade e Performance
    - Automação de Testes
    - Benchmarks e Comparativos

11. Problemas Conhecidos e Soluções
    - Compatibilidade entre Componentes
    - Gestão de Memória e Performance
    - Erros de Serialização e Desserialização
    - Problemas de Treinamento
    - Incompatibilidade de Tokenização

12. Guia para Desenvolvedores
    - Estendendo o LunaGPT
    - Criação de Novos Componentes
    - Práticas de Codificação
    - Contribuindo para o Projeto

13. Referências e Recursos
    - Bibliografia Técnica
    - Recursos de Aprendizagem
    - Comunidade e Suporte

14. Licença

---

## Visão Geral e Fundamentos

### O que é o LunaGPT

O LunaGPT é um sistema de diálogo neural avançado projetado especificamente para interações em português, representando uma evolução significativa em relação aos modelos de linguagem convencionais. Diferentemente de sistemas puramente baseados em transformers, o LunaGPT implementa uma arquitetura híbrida inovadora que combina elementos de transformers com state-space models e mixture of experts, resultando em um modelo altamente eficiente, adaptável e capaz de compreender nuances linguísticas específicas do português.

Desenvolvido com foco na adaptabilidade a diferentes recursos computacionais, o LunaGPT pode operar em ambientes que variam desde servidores de alto desempenho até dispositivos com recursos mais limitados, ajustando automaticamente sua complexidade e consumo de recursos conforme necessário.

### Principais Inovações

O LunaGPT traz diversas inovações tecnológicas que o diferenciam no campo dos modelos de linguagem:

1. **Arquitetura Neural Híbrida**: Integração harmoniosa de transformers, state-space models e mixture of experts em um único sistema coeso, aproveitando os pontos fortes de cada abordagem.

2. **GrowingNetwork**: Implementação pioneira de redes neurais que crescem organicamente durante o treinamento, expandindo sua capacidade apenas quando necessário - uma abordagem que otimiza recursos e melhora a adaptabilidade.

3. **Proatividade Contextual**: Sistema único de detecção de padrões conversacionais que permite ao modelo antecipar necessidades do usuário e oferecer sugestões proativas, elevando a experiência de interação.

4. **Sistema RAG Adaptativo**: Implementação avançada de Retrieval-Augmented Generation com capacidade de fallback para ambientes com recursos limitados, garantindo precisão factual mesmo em hardware modesto.

5. **Curriculum Learning Automatizado**: Sistema de treinamento progressivo que aumenta gradualmente a complexidade dos dados, resultando em modelos mais estáveis e com melhor generalização.

### Diferenciadores Técnicos

O LunaGPT se destaca por uma série de características técnicas diferenciadoras:

1. **Tokenização Especializada para Português**: Tokenizador desenvolvido especificamente para capturar as nuances morfológicas e sintáticas do português, incluindo conjugações verbais complexas, contrações e acentuação.

2. **HyperNetworks para Adaptabilidade**: Utilização de redes neurais secundárias (hypernetworks) que geram dinamicamente parâmetros para as redes principais, permitindo adaptação contextual sem necessidade de fine-tuning tradicional.

3. **Detecção Automática de Hardware**: Sistema sofisticado que identifica as capacidades do hardware disponível e ajusta automaticamente a configuração do modelo, garantindo desempenho otimizado em qualquer ambiente.

4. **Sistema de Feedback Multi-dimensional**: Coleta e processamento de feedback do usuário em múltiplas dimensões (precisão, relevância, tom), permitindo refinamento direcionado e personalizado do modelo.

5. **Personas Configuráveis**: Capacidade de adaptar o estilo, tom e profundidade das respostas através de diferentes personas pré-definidas, sem necessidade de retreinamento do modelo.

### Arquitetura Híbrida

A arquitetura híbrida do LunaGPT representa uma abordagem inovadora na construção de modelos de linguagem:

**Núcleo Transformer**: A base do sistema é construída sobre uma arquitetura transformer, especificamente uma variante do GPT (Generative Pre-trained Transformer), que proporciona excelente modelagem de contexto local e capacidade de geração de texto coerente.

**State-Space Layers**: Complementando o mecanismo de atenção dos transformers, as camadas state-space modelam eficientemente dependências de longo alcance com complexidade computacional linear, permitindo ao modelo capturar padrões distribuídos em sequências muito longas.

**Mixture of Experts**: Em vez de processar toda entrada através de uma única rede densa, o sistema utiliza um conjunto de "redes especialistas", com um mecanismo de roteamento que direciona diferentes entradas para especialistas específicos, aumentando significativamente a capacidade do modelo sem incremento proporcional em custo computacional.

**Integração Harmoniosa**: Estes componentes são integrados de forma a complementarem-se mutuamente, com o transformer capturando dependências locais complexas, as state-space layers modelando eficientemente contexto de longo alcance, e o MoE aumentando a capacidade geral do modelo enquanto mantém eficiência computacional.

---

## Fundamentos Teóricos

### Arquitetura Transformer

A arquitetura Transformer, introduzida no artigo "Attention is All You Need" (Vaswani et al., 2017), revolucionou o processamento de linguagem natural ao eliminar redes recorrentes e convolucionais em favor do mecanismo de auto-atenção.

#### Conceitos Fundamentais:

1. **Mecanismo de Auto-Atenção**: O componente central que permite que cada token na sequência "preste atenção" a todos os outros tokens, atribuindo diferentes pesos a diferentes posições. Esta capacidade de considerar o contexto completo em vez de apenas o contexto local é crucial para compreender nuances linguísticas.

   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k)V
   ```

   Onde:
   - Q (query), K (key), V (value) são projeções lineares da entrada
   - √d_k é um fator de escala para estabilizar os gradientes

2. **Atenção Multi-Cabeça**: Permite que o modelo projete a entrada em múltiplos subespaços de representação diferentes, capturando diversos tipos de relações entre tokens:

   ```
   MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
   where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
   ```

3. **Embeddings Posicionais**: Como o Transformer não possui recorrência ou convolução, não há noção inerente de ordem dos tokens. Os embeddings posicionais adicionam informação sobre a posição dos tokens:

   ```
   PE(pos, 2i) = sin(pos/10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
   ```

4. **Feed-Forward Networks**: Redes densas aplicadas independentemente a cada posição:

   ```
   FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
   ```

#### Implementação no LunaGPT:

No LunaGPT, o transformer base é modificado para incorporar características específicas:

- **Embeddings de Token Aprimorados**: Vetores densos otimizados para representação de tokens em português
- **Escalonamento de Atenção Adaptativo**: Ajuste dinâmico dos fatores de escala da atenção baseado no conteúdo
- **Feed-Forward Especializado**: Substituição seletiva de camadas feed-forward convencionais por blocos MoE
- **Integração com State-Space**: Composição paralela de camadas transformer e state-space para modelagem complementar

### Modelos State-Space

Os State-Space Models (SSMs) representam uma abordagem fundamentalmente diferente para modelagem sequencial, inspirada na teoria de sistemas lineares invariantes no tempo da engenharia de controle.

#### Formulação Matemática:

Em sua forma contínua, um SSM é definido por:

```
x'(t) = Ax(t) + Bu(t)    # Equação de evolução do estado
y(t) = Cx(t) + Du(t)     # Equação de observação
```

Onde:
- x(t) ∈ ℝ^n é o vetor de estado no tempo t
- u(t) ∈ ℝ^d é o vetor de entrada
- y(t) ∈ ℝ^d é o vetor de saída
- A ∈ ℝ^(n×n), B ∈ ℝ^(n×d), C ∈ ℝ^(d×n), D ∈ ℝ^(d×d) são matrizes de parâmetros aprendíveis

#### Discretização para Implementação:

Para aplicações em redes neurais, esta formulação contínua é discretizada:

```
h_t = Ah_{t-1} + Bu_t    # Atualização de estado
y_t = Ch_t + Du_t        # Geração de saída
```

#### Vantagens dos SSMs:

1. **Complexidade Linear**: Ao contrário da atenção (quadrática), os SSMs têm complexidade linear em relação ao comprimento da sequência
2. **Memória de Longo Alcance**: Capacidade de modelar eficientemente dependências temporais muito distantes
3. **Eficiência Computacional**: Menor requisito de memória e computação para sequências longas
4. **Paralelização**: Permite computação paralela eficiente durante treinamento

#### Implementação no LunaGPT:

A `StateSpaceLayer` no LunaGPT implementa um SSM discreto e treinável:

```python
class StateSpaceLayer(nn.Module):
    def __init__(self, hidden_size, state_size=None, activation=None):
        super().__init__()
        self.hidden_size = hidden_size
        # Estado interno menor para eficiência
        self.state_size = state_size or (hidden_size // 4)
        
        # Parâmetros aprendíveis do SSM
        self.A = nn.Parameter(torch.randn(self.state_size, self.state_size) * 0.01)
        self.B = nn.Parameter(torch.randn(self.state_size, self.hidden_size) * 0.01)
        self.C = nn.Parameter(torch.randn(self.hidden_size, self.state_size) * 0.01)
        self.D = nn.Parameter(torch.randn(self.hidden_size) * 0.01)
        
        # Projeções de entrada e ativação
        self.in_proj = nn.Linear(hidden_size, self.state_size)
        self.activation = activation or nn.Tanh()
```

Esta implementação é estrategicamente integrada com camadas de atenção tradicional, permitindo ao modelo combinar o melhor dos dois mundos: modelagem de dependências locais complexas (atenção) e modelagem eficiente de longo alcance (SSM).

### Mixture of Experts

Mixture of Experts (MoE) é uma técnica poderosa para aumentar a capacidade dos modelos neurais sem aumento proporcional nos requisitos computacionais, baseada no princípio "dividir para conquistar" - diferentes partes da entrada são processadas por diferentes sub-redes especializadas.

#### Princípios Fundamentais:

1. **Especialização**: Múltiplas redes (especialistas) são treinadas para se tornarem especialistas em diferentes subconjuntos dos dados.

2. **Roteamento Dinâmico**: Um mecanismo de roteamento (normalmente uma rede neural) decide quais especialistas devem processar cada entrada.

3. **Esparsidade**: Apenas um pequeno subconjunto de especialistas é ativado para cada entrada, criando um processamento esparso que economiza recursos.

#### Formulação Matemática:

Para uma entrada x, a saída do MoE é:

```
y = ∑ G(x)_i × E_i(x)
```

Onde:
- G(x) é o portão/roteador que produz pesos para cada especialista
- E_i é a função de transferência do i-ésimo especialista
- Apenas os top-k especialistas com maior peso são utilizados (esparsidade)

#### Implementação no LunaGPT:

O `MoEBlock` no LunaGPT implementa um sistema MoE sofisticado:

```python
class MoEBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, num_experts=4, sparse_top_k=2, 
                 emotional_routing=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or (4 * input_dim)
        self.num_experts = num_experts
        self.sparse_top_k = sparse_top_k
        self.emotional_routing = emotional_routing
        
        # Criar especialistas - cada um é uma MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Roteador - decide quais especialistas utilizar
        self.router = nn.Linear(input_dim, num_experts)
        
        # Roteamento emocional (opcional)
        if emotional_routing:
            self.emotional_projector = nn.Linear(input_dim, 8)  # 8 dimensões emocionais
            self.emotional_biases = nn.Parameter(torch.randn(8, num_experts) * 0.01)
```

#### Vantagens do MoE no LunaGPT:

1. **Aumento de Capacidade**: Permite aumentar significativamente o número de parâmetros (capacidade) sem aumento proporcional em FLOPs durante inferência.

2. **Eficiência Computacional**: Como apenas um subconjunto de parâmetros é ativado para cada entrada, o custo computacional é mantido sob controle.

3. **Especialização Emergente**: Especialistas naturalmente se especializam em diferentes aspectos da linguagem (domínios, estilos, tipos de raciocínio).

4. **Balanceamento de Carga**: O sistema inclui mecanismos para evitar que alguns especialistas sejam sobrecarregados enquanto outros ficam subutilizados.

5. **Roteamento Emocional**: Extensão única que permite roteamento baseado em características emocionais do texto, melhorando respostas para conteúdo emocionalmente carregado.

### Redes Neurais Adaptativas

O LunaGPT integra vários mecanismos para adaptação neural dinâmica, permitindo que a estrutura e comportamento da rede se ajustem às necessidades computacionais e aos dados:

#### GrowingNetwork: Expansão Neural Orgânica

A `GrowingNetwork` é uma implementação inovadora que permite que a rede cresça organicamente durante o treinamento, adicionando novas camadas quando deteta que atingiu um platô na performance:

```python
class GrowingNetwork(nn.Module):
    """
    Rede que pode crescer incrementalmente durante o treinamento.
    """
    
    def __init__(self, base_model, max_extra_layers=3, growth_threshold=0.001,
                plateau_patience=5):
        super().__init__()
        self.base_model = base_model
        self.max_extra_layers = max_extra_layers
        self.growth_threshold = growth_threshold
        self.plateau_patience = plateau_patience
        
        # Estado de crescimento
        self.extra_layers = nn.ModuleList([])
        self.plateau_counter = 0
        self.best_loss = float('inf')
        
    def add_layer(self, input_dim, output_dim=None):
        """Adiciona nova camada à rede quando necessário."""
        output_dim = output_dim or input_dim
        new_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )
        self.extra_layers.append(new_layer)
        return len(self.extra_layers)
        
    def should_grow(self, current_loss):
        """Determina se a rede deve crescer baseado no histórico de perdas."""
        if current_loss < self.best_loss - self.growth_threshold:
            # Progresso significativo, resetar contador
            self.best_loss = current_loss
            self.plateau_counter = 0
            return False
        else:
            # Possível platô
            self.plateau_counter += 1
            if self.plateau_counter >= self.plateau_patience:
                if len(self.extra_layers) < self.max_extra_layers:
                    self.plateau_counter = 0
                    return True
        return False
```

Esta abordagem traz vários benefícios:
- **Eficiência de Recursos**: O modelo começa mais simples e cresce apenas quando necessário
- **Adaptação à Complexidade**: A capacidade se expande organicamente para atender à complexidade dos dados
- **Evitação de Overfitting**: Reduz o risco de overfitting em datasets menores

#### HyperNetworks: Geração Dinâmica de Parâmetros

As `HyperNetworks` são redes secundárias que geram dinamicamente parâmetros para as redes principais, permitindo adaptação contextual sem necessidade de fine-tuning:

```python
class HyperNetwork(nn.Module):
    """
    Gera parâmetros para outras redes baseado no contexto.
    """
    
    def __init__(self, context_dim, target_dim, hidden_dim=None):
        super().__init__()
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim or context_dim * 2
        
        # MLP para geração de parâmetros
        self.network = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        # Geradores de peso e bias
        self.weight_generator = nn.Linear(self.hidden_dim, target_dim * context_dim)
        self.bias_generator = nn.Linear(self.hidden_dim, target_dim)
        
    def forward(self, context_vector):
        """
        Gera parâmetros baseados no vetor de contexto.
        
        Args:
            context_vector: Tensor representando o contexto atual
            
        Returns:
            weight: Matriz de peso gerada
            bias: Vetor de bias gerado
        """
        h = self.network(context_vector)
        
        # Gerar e remodelar parâmetros
        weight = self.weight_generator(h)
        weight = weight.view(-1, self.target_dim, self.context_dim)
        
        bias = self.bias_generator(h)
        
        return weight, bias
```

Esta tecnologia permite:
- **Adaptação Dinâmica**: O modelo pode adaptar-se a diferentes domínios ou estilos sem fine-tuning
- **Personalização Eficiente**: Gera comportamentos especializados para diferentes contextos
- **Economia de Parâmetros**: Evita a necessidade de múltiplos modelos para diferentes contextos

### Fundamentos de Embeddings e Tokenização

A tokenização e representação de texto são fundamentais para o desempenho dos modelos de linguagem. O LunaGPT implementa várias técnicas avançadas nesta área:

#### Tokenização Especializada para Português

O `LunaTokenizer` implementa um tokenizador BPE (Byte-Pair Encoding) otimizado para português, com tratamento especial para:

1. **Contrações**: Tratamento específico de contrações comuns como "do", "na", "pela", etc.
2. **Conjugações Verbais**: Reconhecimento eficiente de sufixos verbais complexos do português
3. **Acentuação e Diacríticos**: Preservação adequada de informação sobre acentos e outros diacríticos

```python
class LunaTokenizer:
    """
    Tokenizador otimizado para português com tokens especiais para controle de diálogo.
    """
    
    def __init__(self, config):
        self.config = config
        self.max_length = config.tokenizer.get("max_length", 1024)
        
        # Tokens especiais para controle de diálogo
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.system_token = "<|system|>"
        self.thinking_token = "<|thinking|>"
        self.proactive_token = "<|proactive|>"
```

#### Embeddings Avançados

O sistema utiliza várias técnicas avançadas para embeddings:

1. **Embeddings de Token Contextualizados**: Incorporação de informação contextual nos embeddings de tokens

2. **Embeddings Posicionais Aprimorados**: Extensão dos embeddings posicionais tradicionais com modelagem relacional:

   ```python
   class RelationalPositionalEmbedding(nn.Module):
       """
       Embeddings posicionais que modelam relações entre posições.
       """
       
       def __init__(self, max_seq_len, hidden_dim):
           super().__init__()
           self.max_seq_len = max_seq_len
           self.hidden_dim = hidden_dim
           
           # Embeddings absolutos tradicionais
           self.absolute_embeddings = nn.Embedding(max_seq_len, hidden_dim)
           
           # Embeddings para modelar distâncias relativas
           self.relative_distance_embeddings = nn.Embedding(2 * max_seq_len - 1, hidden_dim)
   ```

3. **Embeddings Semânticos para RAG**: Representação semântica densa de documentos e consultas:

   ```python
   def encode_documents(self, documents):
       """
       Codifica documentos em embeddings semânticos para RAG.
       
       Args:
           documents: Lista de documentos textuais
           
       Returns:
           Tensor de embeddings [num_docs, embedding_dim]
       """
       # Usar SentenceTransformers para criar embeddings semânticos
       return self.sentence_transformer.encode(documents, 
                                             convert_to_tensor=True,
                                             normalize_embeddings=True)
   ```

Estas técnicas avançadas de embeddings e tokenização proporcionam ao LunaGPT uma compreensão mais profunda e nuançada do texto em português, contribuindo significativamente para seu desempenho superior em tarefas de diálogo.

---

## Arquitetura Detalhada do Sistema

### Visão Macro: Componentes Principais

O LunaGPT é construído com uma arquitetura modular em camadas que separa claramente as preocupações e permite extensibilidade e manutenibilidade:

```
┌───────────────────────────────────────────────────────────┐
│                     Interface de Usuário                  │
│     (CLI, API Python, Sistemas de Chat Interativo)        │
├───────────────────────────────────────────────────────────┤
│                  Controladores de Sistema                 │
│   (Gerenciamento de Sessão, Roteamento, Configuração)     │
├─────────────────┬─────────────────┬─────────────────┬─────┤
│   Módulo de     │  Sistema RAG    │  Módulo de      │     │
│   Chat          │  (Recuperação)  │  Feedback       │  U  │
│                 │                 │                 │  T  │
├─────────────────┴─────────────────┴─────────────────┤  I  │
│               Núcleo do Modelo Neural                │  L  │
│       (LunaModel, Tokenizador, Componentes)         │  I  │
├─────────────────┬─────────────────┬─────────────────┤  T  │
│   Pipeline de   │  Gestão de      │  Sistema de     │  Á  │
│   Treinamento   │  Dados          │  Avaliação      │  R  │
│                 │                 │                 │  I  │
├─────────────────┴─────────────────┴─────────────────┤  O  │
│               Adaptação de Hardware                 │  S  │
│    (Detecção, Otimização, Quantização Dinâmica)     │     │
└───────────────────────────────────────────────────────────┘
```

Cada componente principal tem responsabilidades específicas:

1. **Interface de Usuário**: Proporciona diferentes pontos de entrada para interação com o sistema
2. **Controladores de Sistema**: Gerenciam fluxo de controle e integração entre componentes
3. **Módulos Funcionais** (Chat, RAG, Feedback): Implementam funcionalidades específicas do sistema
4. **Núcleo Neural**: Encapsula toda a lógica do modelo de linguagem e componentes neurais
5. **Subsistemas de Suporte**: Gerenciam treinamento, dados e avaliação
6. **Adaptação de Hardware**: Otimiza o sistema para diferentes ambientes computacionais

Esta arquitetura modular permite que cada componente evolua de forma independente, facilita testes isolados, e permite substituição ou extensão de funcionalidades específicas sem afetar o restante do sistema.

### Estrutura de Diretórios Completa

O LunaGPT segue uma estrutura de diretórios bem organizada que reflete sua arquitetura modular:

```
LunaGPT/
├── data/                  # Datasets para treinamento e validação
│   ├── train/            # Dados de treinamento em vários formatos
│   │   ├── dialogos/     # Conversas para treinamento supervisonado
│   │   ├── documentos/   # Documentos para treinamento de RAG
│   │   └── instrucoes/   # Dados de instrução para fine-tuning
│   └── valid/            # Dados de validação em vários formatos
├── logs/                  # Registros de execução do sistema
│   ├── training/         # Logs específicos de treinamentos
│   ├── inference/        # Logs de inferência e uso
│   └── errors/           # Registros detalhados de erros
├── models/               # Modelos treinados e checkpoints
│   └── [model_name]/     # Diretório específico para cada modelo
│       ├── components/   # Componentes avançados serializados
│       │   ├── moe/      # Componentes Mixture of Experts
│       │   ├── hypernet/ # HyperNetworks serializados
│       │   └── ssl/      # State-Space Layers serializadas
│       ├── tokenizer/    # Arquivos do tokenizador
│       │   ├── tokenizer.json            # Vocabulário base
│       │   └── special_tokens.json       # Tokens especiais
│       ├── config/       # Configurações específicas do modelo
│       ├── checkpoints/  # Pontos de verificação durante treinamento 
│       └── retriever/    # Base de conhecimento do RAG (quando aplicável)
├── src/                  # Código-fonte principal
│   ├── chat/            # Módulo de interface de conversação
│   │   ├── luna_chat.py            # Sistema de chat principal
│   │   ├── message_formatter.py    # Formatação de mensagens
│   │   ├── session_manager.py      # Gestão de sessões e histórico
│   │   └── proactive_messenger.py  # Sistema de sugestões proativas
│   ├── config/          # Configurações do sistema
│   │   ├── config.py               # Definições de configuração
│   │   └── schema.py               # Esquemas de validação
│   ├── models/          # Definições de arquiteturas neurais
│   │   ├── feedback_system.py      # Sistema de feedback e aprendizado contínuo
│   │   ├── growing_network.py      # Redes expansíveis dinamicamente
│   │   ├── hypernet.py             # HyperNetworks para geração de parâmetros
│   │   ├── luna_model.py           # Modelo principal do sistema
│   │   ├── moe.py                  # Implementação de Mixture of Experts
│   │   ├── state_space_layer.py    # Camada State-Space para sequências
│   │   ├── rag_retriever.py        # Sistema de RAG para busca contextual
│   │   ├── relational_embeddings.py # Embeddings relacionais avançados
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
│   │   ├── curriculum.py           # Implementação de curriculum learning
│   │   ├── loss_functions.py       # Funções de perda personalizadas
│   │   ├── metrics.py              # Métricas de avaliação
│   │   └── trainer.py              # Sistema principal de treinamento
│   └── utils/           # Ferramentas auxiliares
│       ├── data_augmentation.py    # Técnicas de aumento de dados
│       ├── dependency_utils.py     # Verificação e instalação de dependências
│       ├── file_utils.py           # Manipulação de arquivos e formatos
│       ├── hardware_utils.py       # Detecção e otimização para hardware
│       ├── logging_utils.py        # Configuração de logs
│       ├── quantization.py         # Ferramentas de quantização
│       └── visualization.py        # Visualização de treino e resultados
├── temp/                # Arquivos temporários (gerados em execução)
├── wandb/               # Artefatos do Weights & Biases (quando habilitado)
├── .vscode/             # Configurações para VS Code
│   ├── settings.json              # Configuração de ambiente de desenvolvimento
│   └── launch.json                # Configurações de depuração
├── feedback.jsonl       # Banco de dados de feedback de usuários
├── main.py              # Ponto de entrada principal da aplicação
├── setup.py             # Script de instalação e dependências
├── requirements.txt     # Dependências básicas
├── requirements-dev.txt # Dependências para desenvolvimento
├── requirements-gpu.txt # Dependências para aceleração GPU
└── README.md            # Este documento
```

Esta estrutura organizada proporciona várias vantagens:
- **Modularidade**: Cada diretório tem uma responsabilidade clara
- **Escalabilidade**: Fácil adição de novos componentes ou funcionalidades
- **Manutenibilidade**: Separação clara de preocupações
- **Colaboração**: Desenvolvedores podem trabalhar em áreas distintas simultaneamente

### Fluxo de Dados e Processamento

O LunaGPT implementa um fluxo de dados sofisticado desde a entrada do usuário até a geração da resposta final:

#### Fluxo de Inferência (Geração de Respostas)

```
                    ┌───────────────────┐
                    │ Entrada do Usuário │
                    └─────────┬─────────┘
                              ▼
┌─────────────────────────────────────────────────┐
│              Preprocessamento                   │
│  (Normalização, Formatação, Tokenização)        │
└─────────────────────────┬───────────────────────┘
                          ▼
┌─────────────────────────────────────────────────┐
│         Análise de Contexto Conversacional      │
│  (Histórico, Padrões de Diálogo, Intenções)     │
└─────────────────────────┬───────────────────────┘
                          ▼
┌─────────────────┐     ┌─────────────────────────┐
│  Sistema RAG    │◄───►│   Núcleo do Modelo      │
│ (Opcional)      │     │   Neural                │
└─────────────────┘     └───────────┬─────────────┘
                                    ▼
┌─────────────────────────────────────────────────┐
│               Pós-processamento                 │
│  (Formatação, Verificação, Melhorias)           │
└─────────────────────────┬───────────────────────┘
                          ▼
┌─────────────────────────────────────────────────┐
│          Sistema Proativo (Opcional)            │
│  (Detecção de Oportunidades, Sugestões)         │
└─────────────────────────┬───────────────────────┘
                          ▼
                    ┌───────────────────┐
                    │ Resposta Final    │
                    └───────────────────┘
```

**Detalhes das Etapas:**

1. **Preprocessamento**:
   - Normalização de texto (acentuação, espaçamento)
   - Formatação em template de conversação
   - Tokenização para representação numérica

2. **Análise de Contexto Conversacional**:
   - Extração de histórico relevante
   - Identificação de padrões de diálogo
   - Detecção de intenções do usuário

3. **Sistema RAG** (quando ativado):
   - Análise da consulta
   - Recuperação de documentos relevantes
   - Enriquecimento do contexto do modelo

4. **Processamento pelo Núcleo Neural**:
   - Geração de embeddings de entrada
   - Processamento através das camadas neurais
   - Etapa de "thinking" (raciocínio interno)
   - Geração da resposta base

5. **Pós-processamento**:
   - Formatação da resposta
   - Verificações de qualidade e segurança
   - Melhorias estilísticas (quando configuradas)

6. **Sistema Proativo** (quando ativado):
   - Análise de oportunidades para sugestões proativas
   - Geração de sugestões contextuais
   - Integração com a resposta principal

Este fluxo de processamento multi-estágio garante respostas de alta qualidade, contextualmente relevantes e adaptadas às necessidades do usuário.

#### Fluxo de Treinamento

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Dados Brutos  │───►│ Processamento │───►│  Dataset      │
│               │    │ de Dados      │    │  Formatado    │
└───────────────┘    └───────────────┘    └───────┬───────┘
                                                  │
                                                  ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Configuração  │───►│ Inicialização │◄───┤  Curriculum   │
│ de Treino     │    │ do Modelo     │    │  Learning     │
└───────────────┘    └───────┬───────┘    └───────────────┘
                            │
                            ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Callbacks    │◄───┤ Loop de       │───►│  Validação    │
│               │    │ Treinamento   │    │  Periódica    │
└───────────────┘    └───────┬───────┘    └───────────────┘
                            │
                            ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Checkpoints   │◄───┤ Avaliação     │───►│  Métricas     │
│               │    │ Final         │    │  Detalhadas   │
└───────────────┘    └───────┬───────┘    └───────────────┘
                            │
                            ▼
                     ┌───────────────┐
                     │ Modelo        │
                     │ Treinado      │
                     └───────────────┘
```

Este fluxo de treinamento inclui várias otimizações avançadas:

- **Curriculum Learning**: Aumento progressivo da complexidade dos dados
- **Adaptação Dinâmica**: Ajuste de hiperparâmetros baseado em recursos disponíveis
- **Validação Estratégica**: Validação periódica com métricas personalizadas
- **Crescimento Neural**: Adição de camadas durante treinamento quando necessário
- **Monitoramento Avançado**: Integração com sistemas de rastreamento de experimentos

---

## Componentes Técnicos Nucleares

### LunaModel: O Coração do Sistema

O `LunaModel` é o componente central do sistema, integrando todas as inovações arquiteturais e servindo como interface principal para processamento de linguagem:

```python
class LunaModel:
    """
    Modelo neural principal do LunaGPT, integrando transformer com componentes avançados.
    
    Implementa uma arquitetura híbrida que combina elementos de:
    - Transformer tradicional (para modelagem de contexto local)
    - State-Space Models (para dependências de longo alcance)
    - Mixture of Experts (para aumento de capacidade com eficiência)
    - HyperNetworks (para adaptabilidade dinâmica)
    - GrowingNetworks (para crescimento orgânico durante treinamento)
    """
    
    def __init__(self, config, base_model=None):
        """
        Inicializa o modelo Luna.
        
        Args:
            config: Configuração do modelo
            base_model: Modelo base pré-inicializado (opcional)
        """
        self.config = config
        self.device = config.device if hasattr(config, "device") else "cpu"
        
        # Componentes arquiteturais
        self.base_transformer = base_model
        self.moe_blocks = None
        self.state_space_layers = None
        self.hyper_networks = None
        self.growing_network = None
        
        # Indicadores de recursos
        self.use_moe = config.use_moe
        self.use_state_space = config.use_state_space
        self.use_hypernet = config.use_hypernet
        self.use_growing = config.use_growing
        
        # Inicializar ou configurar modelo base
        if self.base_transformer is None:
            self._initialize_base_model()
            
        # Inicializar componentes avançados
        if self.use_moe:
            self._initialize_moe()
        if self.use_state_space:
            self._initialize_state_space()
        if self.use_hypernet:
            self._initialize_hypernet()
        if self.use_growing:
            self._initialize_growing()
            
        # Mover para dispositivo adequado
        self._prepare_device()
```

#### Métodos Principais

```python
def from_scratch(cls, config, use_lightweight=False):
    """
    Cria um modelo do zero.
    
    Args:
        config: Configuração do modelo
        use_lightweight: Se deve usar configuração leve para hardware limitado
        
    Returns:
        Instância de LunaModel inicializada
    """
    
def from_pretrained(cls, model_path, config=None):
    """
    Carrega modelo de checkpoint existente.
    
    Args:
        model_path: Caminho para o diretório do modelo
        config: Configuração opcional (extraída do modelo se None)
        
    Returns:
        Instância de LunaModel carregada do checkpoint
    """
    
def generate(self, input_ids, attention_mask=None, max_length=100, **kwargs):
    """
    Gera texto baseado na entrada.
    
    Args:
        input_ids: IDs de tokens de entrada
        attention_mask: Máscara de atenção (opcional)
        max_length: Comprimento máximo da sequência gerada
        **kwargs: Argumentos adicionais de geração
        
    Returns:
        Tensor de tokens gerados
    """
    
def save(self, model_dir, save_to_wandb=False, run_name=None):
    """
    Salva o modelo e seus componentes.
    
    Args:
        model_dir: Diretório para salvar o modelo
        save_to_wandb: Se deve salvar também no W&B
        run_name: Nome da execução W&B (se aplicável)
    """
```

#### Características Arquiteturais Distintivas

1. **Integração Transparente de Componentes**: O `LunaModel` integra componentes heterogêneos de forma transparente, permitindo que funcionem harmonicamente

2. **Checkpoint Inteligente**: Sistema sofisticado de checkpoint que preserva todos os componentes avançados de forma reconstituível:
    ```python
    def _save_components(self, components_dir):
        """Salva componentes arquiteturais avançados."""
        os.makedirs(components_dir, exist_ok=True)
        
        # Salvar MoE
        if self.moe_blocks:
            moe_dir = os.path.join(components_dir, "moe")
            os.makedirs(moe_dir, exist_ok=True)
            for i, block in enumerate(self.moe_blocks):
                torch.save(block, os.path.join(moe_dir, f"moe_block_{i}.pt"))
                # Salvar também state_dict para compatibilidade
                torch.save(block.state_dict(), os.path.join(moe_dir, f"moe_block_{i}_state_dict.pt"))
                
        # Salvar State-Space Layers
        if self.state_space_layers:
            ssl_dir = os.path.join(components_dir, "ssl")
            os.makedirs(ssl_dir, exist_ok=True)
            for i, layer in enumerate(self.state_space_layers):
                torch.save(layer, os.path.join(ssl_dir, f"ssl_{i}.pt"))
                torch.save(layer.state_dict(), os.path.join(ssl_dir, f"ssl_{i}_state_dict.pt"))
    ```

3. **Capacidades de Adaptação Dinâmica**:
   - Detecção e otimização para hardware disponível
   - Chaveamento dinâmico entre modos de execução
   - Auto-ajuste de parâmetros baseado em contexto

4. **Pipeline de Geração Avançado**:
   ```python
   def _process_generation_with_components(self, base_output, **kwargs):
       """
       Enriquece a geração base com processamento dos componentes avançados.
       
       Esta função implementa o pipeline completo de geração que inclui:
       1. Geração inicial pelo transformer base
       2. Processamento pelo State-Space Model para contexto de longo alcance
       3. Roteamento através de especialistas MoE para domínios específicos
       4. Adaptação final via HyperNetwork para ajuste contextual
       """
   ```

### Mixture of Experts (MoEBlock)

O componente `MoEBlock` implementa o padrão Mixture of Experts para aumentar a capacidade do modelo com eficiência computacional:

```python
class MoEBlock(nn.Module):
    """
    Implementação avançada de Mixture of Experts com roteamento adaptativo.
    
    Permite que o modelo mantenha grande capacidade (muitos parâmetros) enquanto
    ativa seletivamente apenas um subconjunto para cada entrada, resultando em
    eficiência computacional.
    """
    
    def __init__(self, input_dim, hidden_dim=None, num_experts=4, sparse_top_k=2,
                 expert_dropout=0.1, load_balancing=True, emotional_routing=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or (4 * input_dim)
        self.num_experts = num_experts
        self.sparse_top_k = min(sparse_top_k, num_experts)  # Garantir k <= num_experts
        self.load_balancing = load_balancing
        self.emotional_routing = emotional_routing
        
        # Criar especialistas (redes feed-forward)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(expert_dropout),
                nn.Linear(self.hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Roteador - decide quais especialistas usar para cada token
        self.router = nn.Linear(input_dim, num_experts)
        
        # Componente de roteamento emocional (opcional)
        if emotional_routing:
            # Detector de emoções - dimensões: alegria, tristeza, raiva, medo, etc.
            self.emotional_dim = 8
            self.emotional_projector = nn.Linear(input_dim, self.emotional_dim)
            self.emotional_biases = nn.Parameter(torch.randn(self.emotional_dim, num_experts) * 0.01)
            
        # Contador para balanceamento de carga
        register_buffer("expert_counts", torch.zeros(num_experts))
```

#### Fluxo de Roteamento

O coração do `MoEBlock` é o mecanismo de roteamento que decide quais especialistas processar cada token:

```python
def forward(self, x):
    """
    Processa entrada através dos especialistas apropriados.
    
    Args:
        x: Tensor de entrada [batch_size, seq_len, input_dim]
        
    Returns:
        Tensor processado [batch_size, seq_len, input_dim]
    """
    batch_size, seq_len, input_dim = x.shape
    
    # Remodelar para processamento por token
    x_flat = x.reshape(-1, input_dim)  # [batch_size*seq_len, input_dim]
    
    # Obter scores de roteamento
    routing_logits = self.router(x_flat)  # [batch_size*seq_len, num_experts]
    
    # Adicionar componente emocional se habilitado
    if self.emotional_routing:
        emotional_features = self.emotional_projector(x_flat)  # [batch_size*seq_len, emotional_dim]
        emotional_routing = torch.matmul(emotional_features, self.emotional_biases)  # [batch_size*seq_len, num_experts]
        routing_logits = routing_logits + emotional_routing
    
    # Aplicar roteamento esparso (top-k)
    routing_top_k_values, routing_top_k_indices = torch.topk(
        routing_logits, k=self.sparse_top_k, dim=-1, largest=True, sorted=True
    )
    
    # Criar máscara de roteamento esparsa
    routing_top_k_mask = torch.zeros_like(routing_logits).scatter_(
        -1, routing_top_k_indices, 1.0
    )
    
    # Normalizar pesos (apenas para especialistas ativos)
    routing_weights = torch.softmax(routing_logits + (1 - routing_top_k_mask) * -1e10, dim=-1)
    
    # Processar através de especialistas ativos
    expert_outputs = torch.zeros_like(x_flat)
    for i in range(self.num_experts):
        # Máscara para tokens que usam este especialista
        expert_mask = routing_weights[:, i:i+1]
        
        # Aplicar máscara e processar pelo especialista
        if expert_mask.sum() > 0:
            expert_inputs = x_flat * expert_mask
            expert_output = self.experts[i](expert_inputs)
            expert_outputs += expert_output
            
            # Atualizar contadores para balanceamento
            if self.training and self.load_balancing:
                self.expert_counts[i] += expert_mask.sum().item()
                
    # Aplicar penalidade de balanceamento de carga (durante treinamento)
    if self.training and self.load_balancing:
        # Normalizar contagens
        normalized_counts = self.expert_counts / self.expert_counts.sum()
        # Penalidade para distribuição desigual
        load_balancing_loss = torch.sum(normalized_counts * torch.log(normalized_counts + 1e-10)) * self.num_experts
        # Registrar para uso no treinamento
        self.load_balancing_loss = load_balancing_loss
    
    # Remodelar saída para formato original
    output = expert_outputs.reshape(batch_size, seq_len, input_dim)
    return output
```

#### Características Avançadas

1. **Roteamento Emocional**: Capacidade única de direcionar entradas para especialistas baseado em características emocionais detectadas:
   ```python
   if self.emotional_routing:
       emotional_features = self.emotional_projector(x_flat)  # Detectar emoções
       emotional_routing = torch.matmul(emotional_features, self.emotional_biases)
       routing_logits = routing_logits + emotional_routing  # Influenciar roteamento
   ```

2. **Balanceamento de Carga**: Mecanismo para evitar sobrecarga de alguns especialistas:
   ```python
   # Durante treinamento
   if self.training and self.load_balancing:
       normalized_counts = self.expert_counts / self.expert_counts.sum()
       load_balancing_loss = torch.sum(normalized_counts * torch.log(normalized_counts + 1e-10)) * self.num_experts
   ```
3. **Optimização de Esparsidade**: Implementação eficiente de ativação top-k que ativa apenas um subconjunto de especialistas para cada token:
   ```python
   # Aplicar roteamento esparso (top-k)
   routing_top_k_values, routing_top_k_indices = torch.topk(
       routing_logits, k=self.sparse_top_k, dim=-1, largest=True, sorted=True
   )
   
   # Criar máscara de roteamento esparsa
   routing_top_k_mask = torch.zeros_like(routing_logits).scatter_(
       -1, routing_top_k_indices, 1.0
   )
   ```

4. **Distribuição Dinâmica**: Ajuste automático da carga entre especialistas disponíveis em diferentes condições de hardware:
   ```python
   def adapt_to_hardware(self, hardware_info):
       """
       Ajusta a estratégia de roteamento baseado em recursos disponíveis.
       
       Args:
           hardware_info: Dict com informações sobre recursos disponíveis
       """
       available_memory = hardware_info.get("available_memory_mb", 4000)
       
       # Ajustar grau de esparsidade de acordo com memória
       if available_memory < 2000:  # Recursos limitados
           self.sparse_top_k = 1  # Ultra esparso, cada token usa apenas 1 especialista
       elif available_memory < 6000:  # Recursos moderados
           self.sparse_top_k = min(2, self.num_experts)  # Esparso
       else:  # Recursos abundantes
           self.sparse_top_k = min(3, self.num_experts)  # Menos esparso
   ```

### State-Space Layer: Modelagem de Sequência Eficiente

O componente `StateSpaceLayer` implementa um modelo de espaço de estados para processamento eficiente de sequências longas:

```python
class StateSpaceLayer(nn.Module):
    """
    Camada de espaço de estados para modelagem eficiente de dependências de longo alcance.
    
    Implementa um sistema linear invariante no tempo discreto para processar sequências.
    Comparado com mecanismos de atenção (O(n²)), este componente tem complexidade linear (O(n))
    em relação ao comprimento da sequência, tornando-o ideal para contextos muito longos.
    """
    
    def __init__(self, hidden_size, state_size=None, activation=None, initializer="orthogonal"):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size or (hidden_size // 4)
        
        # Parâmetros aprendíveis do SSM
        if initializer == "orthogonal":
            # Inicialização ortogonal para estabilidade
            self.A = nn.Parameter(torch.empty(self.state_size, self.state_size))
            nn.init.orthogonal_(self.A, gain=0.8)  # Gain < 1 para estabilidade
        else:
            self.A = nn.Parameter(torch.randn(self.state_size, self.state_size) * 0.01)
            
        self.B = nn.Parameter(torch.randn(self.state_size, hidden_size) * 0.01)
        self.C = nn.Parameter(torch.randn(hidden_size, self.state_size) * 0.01)
        self.D = nn.Parameter(torch.zeros(hidden_size))
        
        # Projeção e ativação
        self.activation = activation or nn.Tanh()
        
        # Cache para estado entre chamadas
        self.register_buffer("cached_state", torch.zeros(1, 1, self.state_size))
        self.has_cached_state = False
```

#### Mecanismo de Propagação

O núcleo da camada SSM é o mecanismo de propagação que processa tokens sequencialmente:

```python
def forward(self, x, use_cache=False):
    """
    Processa entrada sequencialmente por um sistema linear.
    
    Args:
        x: Tensor de entrada [batch_size, seq_len, hidden_size]
        use_cache: Se deve usar estado em cache para geração contínua
        
    Returns:
        Tensor processado [batch_size, seq_len, hidden_size]
    """
    batch_size, seq_len, _ = x.shape
    
    # Inicializar estado
    if use_cache and self.has_cached_state:
        h = self.cached_state
    else:
        h = torch.zeros(batch_size, 1, self.state_size, device=x.device)
    
    outputs = []
    
    # Processar sequência token por token (implementação simplificada)
    for t in range(seq_len):
        # Equações do sistema linear discreto
        u = x[:, t:t+1, :]  # Token atual
        h = torch.tanh(h @ self.A.T + u @ self.B.T)  # Atualização de estado
        y = h @ self.C.T + u * self.D  # Saída
        outputs.append(y)
    
    # Armazenar estado final para uso futuro (geração)
    if use_cache:
        self.cached_state = h.detach()
        self.has_cached_state = True
    
    # Concatenar saídas
    return torch.cat(outputs, dim=1)
```

#### Implementação Paralela Eficiente

Para treinamento eficiente, o LunaGPT implementa também uma versão paralela que processa toda a sequência de uma vez:

```python
def parallel_forward(self, x):
    """
    Versão paralela da propagação para treinamento eficiente.
    
    Esta implementação processa toda a sequência de uma vez,
    aproveitando a paralelização para acelerar o treinamento.
    
    Args:
        x: Tensor de entrada [batch_size, seq_len, hidden_size]
        
    Returns:
        Tensor processado [batch_size, seq_len, hidden_size]
    """
    batch_size, seq_len, _ = x.shape
    
    # Precalcular potências da matriz A (convolução)
    A_powers = [torch.eye(self.state_size, device=x.device)]
    for t in range(1, seq_len):
        A_powers.append(A_powers[-1] @ self.A)
    A_powers = torch.stack(A_powers, dim=0)  # [seq_len, state_size, state_size]
    
    # Projeção de entrada
    u = x @ self.B.T  # [batch_size, seq_len, state_size]
    
    # Computar estados para todos os passos de tempo
    h = torch.zeros(batch_size, seq_len, self.state_size, device=x.device)
    for t in range(seq_len):
        for tau in range(t+1):
            h[:, t] += A_powers[t-tau] @ u[:, tau]
    
    # Computar saídas
    y = self.activation(h) @ self.C.T + x * self.D
    
    return y
```

#### Características Técnicas Avançadas

1. **Inicialização Estabilizada**: Uso de inicialização ortogonal com ganho controlado para garantir estabilidade do sistema:
   ```python
   if initializer == "orthogonal":
       self.A = nn.Parameter(torch.empty(self.state_size, self.state_size))
       nn.init.orthogonal_(self.A, gain=0.8)  # Gain < 1 para estabilidade
   ```

2. **Cache de Estado**: Mecanismo de armazenamento de estado entre chamadas para geração contínua:
   ```python
   # Armazenar estado final para uso futuro (geração)
   if use_cache:
       self.cached_state = h.detach()
       self.has_cached_state = True
   ```

3. **Integração com Transformer**: Mecanismos para integração harmoniosa com camadas transformer:
   ```python
   class HybridBlock(nn.Module):
       """Bloco que combina transformer com state-space."""
       
       def __init__(self, hidden_size, num_heads, dropout=0.1):
           super().__init__()
           self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout)
           self.ssm = StateSpaceLayer(hidden_size)
           self.integration = nn.Linear(hidden_size * 2, hidden_size)
           self.norm = nn.LayerNorm(hidden_size)
           
       def forward(self, x, attention_mask=None):
           # Processamento paralelo
           attn_out, _ = self.attention(x, x, x, attn_mask=attention_mask)
           ssm_out = self.ssm(x)
           
           # Integração adaptativa
           combined = torch.cat([attn_out, ssm_out], dim=-1)
           output = x + self.integration(combined)
           return self.norm(output)
   ```

4. **Adaptação Automática**: Ajuste automático de dimensionalidade baseado em recursos:
   ```python
   def adapt_to_hardware(self, available_memory_mb):
       """Ajusta dimensão de estado conforme recursos disponíveis."""
       if available_memory_mb < 2000:
           self.state_size = max(self.hidden_size // 8, 16)
       elif available_memory_mb < 6000:
           self.state_size = max(self.hidden_size // 4, 32)
   ```

### HyperNetworks: Geração Dinâmica de Parâmetros

A `HyperNetwork` é uma rede que gera parâmetros para outras redes, permitindo adaptação dinâmica a diferentes contextos:

```python
class HyperNetwork(nn.Module):
    """
    Rede que gera parâmetros para outras redes de forma dinâmica.
    
    Esta técnica permite adaptabilidade contextual sem necessidade de fine-tuning
    tradicional, gerando parâmetros adaptados ao contexto atual.
    """
    
    def __init__(self, context_dim, target_dim, hidden_dim=None, 
                 activation=nn.GELU, layer_norm=True):
        super().__init__()
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim or (context_dim * 2)
        self.activation = activation()
        self.use_layer_norm = layer_norm
        
        # Rede principal para processamento de contexto
        layers = [nn.Linear(context_dim, self.hidden_dim), self.activation]
        if layer_norm:
            layers.append(nn.LayerNorm(self.hidden_dim))
            
        layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), self.activation])
        if layer_norm:
            layers.append(nn.LayerNorm(self.hidden_dim))
            
        self.network = nn.Sequential(*layers)
        
        # Geradores especializados para pesos e bias
        weight_dim = target_dim * context_dim
        self.weight_generator = nn.Linear(self.hidden_dim, weight_dim)
        self.bias_generator = nn.Linear(self.hidden_dim, target_dim)
        
        # Inicialização especial para geradores
        nn.init.zeros_(self.weight_generator.bias)
        nn.init.zeros_(self.bias_generator.bias)
        
    def forward(self, context_vector):
        """
        Gera parâmetros baseados no vetor de contexto.
        
        Args:
            context_vector: Tensor representando o contexto [batch_size, context_dim]
            
        Returns:
            weight: Matriz de peso gerada [batch_size, target_dim, context_dim]
            bias: Vetor de bias gerado [batch_size, target_dim]
        """
        h = self.network(context_vector)
        
        # Gerar parâmetros
        weight = self.weight_generator(h)
        weight = weight.view(-1, self.target_dim, self.context_dim)
        
        bias = self.bias_generator(h)
        
        return weight, bias
```

#### Aplicações Práticas

A HyperNetwork é aplicada em diversos cenários no LunaGPT:

```python
class HyperAdapter(nn.Module):
    """
    Adaptador que usa HyperNetwork para ajustar comportamento de camadas.
    """
    def __init__(self, input_dim, meta_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.meta_dim = meta_dim
        
        # Extrator de meta-informação
        self.meta_extractor = nn.Linear(input_dim, meta_dim)
        
        # HyperNetwork que gera parâmetros do adaptador
        self.hypernet = HyperNetwork(meta_dim, input_dim)
        
    def forward(self, x):
        # Extrair meta-informação do input
        meta_features = self.meta_extractor(x.mean(dim=1))  # Pooling
        
        # Gerar parâmetros adaptativos
        weight, bias = self.hypernet(meta_features)
        
        # Aplicar transformação adaptativa
        batch_size = x.shape[0]
        adapted_outputs = []
        
        for i in range(batch_size):
            # Multiplicação matriz-vetor para cada exemplo na batch
            adapted = torch.matmul(weight[i], x[i].T).T + bias[i]
            adapted_outputs.append(adapted)
            
        return torch.stack(adapted_outputs)
```

#### Aplicações Específicas por Domínio

```python
class DomainSpecificHyperAdapter(nn.Module):
    """
    Adaptador que especializa o comportamento do modelo para domínios específicos.
    
    Usa HyperNetworks para detectar o domínio do input e gerar parâmetros
    específicos para esse domínio.
    """
    
    def __init__(self, input_dim, num_domains=5, domain_emb_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.num_domains = num_domains
        self.domain_emb_dim = domain_emb_dim
        
        # Detector de domínio
        self.domain_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, num_domains)
        )
        
        # Embeddings de domínio
        self.domain_embeddings = nn.Parameter(
            torch.randn(num_domains, domain_emb_dim)
        )
        
        # HyperNetwork para parâmetros específicos de domínio
        self.hypernet = HyperNetwork(domain_emb_dim, input_dim)
        
    def forward(self, x):
        # Detectar distribuição de domínio
        domain_logits = self.domain_detector(x.mean(dim=1))
        domain_probs = torch.softmax(domain_logits, dim=-1)
        
        # Computar embedding de domínio ponderado
        domain_embedding = torch.matmul(domain_probs, self.domain_embeddings)
        
        # Gerar parâmetros adaptados ao domínio
        weight, bias = self.hypernet(domain_embedding)
        
        # Aplicar transformação específica para o domínio
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            out = torch.matmul(weight[i], x[i].T).T + bias[i]
            outputs.append(out)
            
        return torch.stack(outputs), domain_probs
```

### GrowingNetwork: Expansão Neural Orgânica

A `GrowingNetwork` implementa o conceito inovador de redes que crescem organicamente durante o treinamento:

```python
class GrowingNetwork(nn.Module):
    """
    Rede neural que cresce organicamente durante o treinamento.
    
    Monitora o progresso da perda e adiciona novas camadas quando detecta
    que o modelo atingiu um platô, permitindo aumento gradual e eficiente
    de capacidade.
    """
    
    def __init__(self, base_model, max_extra_layers=3, growth_threshold=0.001,
                plateau_patience=5, hidden_expansion=2.0):
        super().__init__()
        self.base_model = base_model
        self.max_extra_layers = max_extra_layers
        self.growth_threshold = growth_threshold
        self.plateau_patience = plateau_patience
        self.hidden_expansion = hidden_expansion
        
        # Estado de crescimento
        self.extra_layers = nn.ModuleList([])
        self.plateau_counter = 0
        self.best_loss = float('inf')
        self.growth_history = []
        
    def forward(self, x, **kwargs):
        """
        Processa a entrada através do modelo base e camadas extras.
        
        Args:
            x: Tensor de entrada
            **kwargs: Argumentos adicionais para o modelo base
            
        Returns:
            Saída processada por todas as camadas
        """
        # Processar pelo modelo base
        outputs = self.base_model(x, **kwargs)
        
        # Processar por camadas extras (se existirem)
        if len(self.extra_layers) > 0:
            for layer in self.extra_layers:
                if isinstance(outputs, tuple):
                    primary_output = outputs[0]
                    extra_outputs = outputs[1:]
                    primary_output = layer(primary_output)
                    outputs = (primary_output,) + extra_outputs
                else:
                    outputs = layer(outputs)
                    
        return outputs
        
    def add_layer(self, input_dim, output_dim=None):
        """
        Adiciona nova camada à rede.
        
        Args:
            input_dim: Dimensão de entrada da nova camada
            output_dim: Dimensão de saída (igual à entrada se None)
            
        Returns:
            Índice da nova camada
        """
        output_dim = output_dim or input_dim
        hidden_dim = int(input_dim * self.hidden_expansion)
        
        new_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.1)
        )
        
        # Inicialização próxima da identidade para transição suave
        nn.init.eye_(new_layer[0].weight)
        nn.init.zeros_(new_layer[0].bias)
        nn.init.eye_(new_layer[3].weight)
        nn.init.zeros_(new_layer[3].bias)
        
        self.extra_layers.append(new_layer)
        self.growth_history.append({
            'epoch': self.epoch_counter,
            'loss_before_growth': self.best_loss
        })
        
        return len(self.extra_layers) - 1
        
    def should_grow(self, current_loss):
        """
        Determina se a rede deve crescer baseado no histórico de perdas.
        
        Args:
            current_loss: Valor de perda atual
            
        Returns:
            Boolean indicando se deve crescer
        """
        if current_loss < self.best_loss - self.growth_threshold:
            # Progresso significativo, resetar contador
            self.best_loss = current_loss
            self.plateau_counter = 0
            return False
        elif current_loss < self.best_loss:
            # Progresso pequeno
            self.best_loss = current_loss
            return False
        else:
            # Possível platô
            self.plateau_counter += 1
            if self.plateau_counter >= self.plateau_patience:
                # Verificar se ainda pode crescer
                if len(self.extra_layers) < self.max_extra_layers:
                    self.plateau_counter = 0
                    return True
                
        return False
```

#### Benefícios da Abordagem

1. **Eficiência de Recursos**: O modelo começa pequeno e simples, crescendo apenas quando necessário
2. **Adaptação à Complexidade do Problema**: A capacidade se expande organicamente para atender à complexidade dos dados
3. **Prevenção de Overfitting**: Reduz o risco de overfitting em datasets menores ao começar com modelos mais simples
4. **Treinamento mais Estável**: Permite convergência mais estável ao adicionar complexidade gradualmente
5. **Economia de Recursos Computacionais**: Usa apenas a capacidade necessária para o problema em questão

### Tokenização Especializada para Português

O `LunaTokenizer` é um componente crítico que implementa tokenização especializada para o idioma português:

```python
class LunaTokenizer:
    """
    Tokenizador otimizado para português com tratamento especial para 
    estruturas linguísticas específicas do idioma.
    
    Implementa um modelo BPE (Byte-Pair Encoding) treinado em corpus
    português diversificado com tratamentos específicos para:
    - Contrações (do, na, pelos, etc)
    - Conjugações verbais complexas
    - Acentuação e diacríticos
    - Expressões idiomáticas
    """
    
    def __init__(self, config):
        self.config = config
        self.max_length = config.tokenizer.get("max_length", 1024)
        self.truncation_strategy = config.tokenizer.get("truncation_strategy", "right")
        
        # Tokens especiais para controle de diálogo
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.system_token = "<|system|>"
        self.thinking_token = "<|thinking|>"
        self.proactive_token = "<|proactive|>"
        self.end_token = "<|end|>"
        
        # Carregar tokenizador base
        vocab_path = config.tokenizer.get("vocab_path")
        merges_path = config.tokenizer.get("merges_path")
        
        if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
            raise ValueError(f"Arquivos de vocabulário ou merges não encontrados: {vocab_path}, {merges_path}")
            
        # Inicializar tokenizador BPE base
        self.bpe_tokenizer = Tokenizer(BPE(vocab_path, merges_path))
        
        # Adicionar tokens especiais
        self.special_tokens = [
            self.user_token, self.assistant_token, self.system_token,
            self.thinking_token, self.proactive_token, self.end_token
        ]
        
        # Expor IDs para tokens especiais
        self._add_special_token_ids()
        
        # Adicionar normalizadores específicos para português
        self._add_portuguese_normalizers()
```

#### Tratamentos Específicos para Português

```python
def _add_portuguese_normalizers(self):
    """
    Adiciona normalizadores específicos para o português.
    """
    # Preservar acentuação e cedilha
    self.bpe_tokenizer.normalizer = Sequence([
        NFKC(),
        Replace(r'[^\p{L}\p{N}\p{Z}\p{P}\p{S}\p{M}]', ''),
        # Não converte acentos e cedilha para formas decompostas
        # Normaliza espaços em branco
        Replace(r'\s+', ' ')
    ])
    
    # Adicionar tratamento especial para contrações
    self.contraction_patterns = {
        r'\b([Dd])([aeiou]s?)\b': r'\1\2',  # Contrações com "de": do, da, dos, das
        r'\b([Nn])([aeiou]s?)\b': r'\1\2',  # Contrações com "em": no, na, nos, nas
        r'\b([Aa])([aeiou]s?)\b': r'\1\2',  # Contrações com "a": ao, aos
        r'\b([Pp])([aeiou][lr][aeiou]s?)\b': r'\1\2'  # pelo, pela, etc
    }
    
    # Adicionar tratamento para verbos pronominais
    self.pronominal_patterns = [
        r'(\w+[aeiou]r?)-([mnts]e)\b',  # Ex: fala-se, deu-me
        r'(\w+[aeiou]r?)-([nlv]os?)\b',  # Ex: deu-lhe, fala-nos
    ]
```

#### Métodos de Tokenização e Detokenização

```python
def encode(self, text, add_special_tokens=True, return_tensors=None):
    """
    Tokeniza o texto para IDs de token.
    
    Args:
        text: Texto a ser tokenizado
        add_special_tokens: Se deve adicionar tokens especiais
        return_tensors: Formato de retorno ("pt" para PyTorch, None para lista)
        
    Returns:
        Sequência de IDs de token
    """
    # Preprocessamento específico para português
    text = self._preprocess_portuguese(text)
    
    # Tokenização principal
    encoded = self.bpe_tokenizer.encode(text)
    token_ids = encoded.ids
    
    # Adicionar tokens especiais conforme necessário
    if add_special_tokens:
        # Lógica para tokens especiais
        pass
        
    # Truncar se necessário
    if len(token_ids) > self.max_length:
        if self.truncation_strategy == "right":
            token_ids = token_ids[:self.max_length]
        else:  # "left"
            token_ids = token_ids[-self.max_length:]
            
    # Converter para tensor se solicitado
    if return_tensors == "pt":
        return torch.tensor([token_ids])
    
    return token_ids
    
def decode(self, token_ids, skip_special_tokens=True):
    """
    Converte IDs de token de volta para texto.
    
    Args:
        token_ids: Lista ou tensor de IDs de token
        skip_special_tokens: Se deve ignorar tokens especiais
        
    Returns:
        Texto decodificado
    """
    # Converter tensor para lista se necessário
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    # Filtrar tokens especiais se solicitado
    if skip_special_tokens:
        token_ids = [id for id in token_ids if id not in self.special_token_ids]
    
    # Decodificação principal
    text = self.bpe_tokenizer.decode(token_ids)
    
    # Pós-processamento para português
    text = self._postprocess_portuguese(text)
    
    return text
```

#### Métodos para Formatação de Diálogo

```python
def format_dialog(self, messages, include_thinking=False):
    """
    Formata uma lista de mensagens de diálogo para entrada do modelo.
    
    Args:
        messages: Lista de dicionários com campo 'role' e 'content'
        include_thinking: Se deve incluir seção de thinking para o assistant
        
    Returns:
        Texto formatado para entrada do modelo
    """
    formatted = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            formatted.append(f"{self.user_token} {content}")
        elif role == "system":
            formatted.append(f"{self.system_token} {content}")
        elif role == "assistant":
            assistant_msg = f"{self.assistant_token} {content}"
            if include_thinking and "thinking" in msg:
                assistant_msg = f"{self.thinking_token} {msg['thinking']} {assistant_msg}"
            formatted.append(assistant_msg)
        elif role == "proactive":
            formatted.append(f"{self.proactive_token} {content}")
    
    return " ".join(formatted) + f" {self.end_token}"
```

---

## Funcionalidades Avançadas

### Sistema RAG (Retrieval-Augmented Generation)

O Sistema RAG do LunaGPT aprimora a geração de texto com informações recuperadas de uma base de conhecimento:

```python
class RAGSystem:
    """
    Sistema de Retrieval-Augmented Generation.
    
    Aprimora as respostas do modelo com informações relevantes
    recuperadas de fontes externas.
    """
    
    def __init__(self, config, embedding_model=None):
        self.config = config
        self.k_retrieval = config.rag.get("k_retrieval", 5)
        self.threshold = config.rag.get("relevance_threshold", 0.7)
        self.max_context_length = config.rag.get("max_context_length", 1500)
        
        # Inicializar modelo de embeddings
        self.embedding_model = embedding_model or self._initialize_embedding_model()
        
        # Inicializar base de documentos
        self.document_store = DocumentStore()
        
        # Inicializar fallback para hardware limitado
        self.fallback_enabled = config.rag.get("enable_fallback", True)
        self.fallback_index = self._initialize_fallback() if self.fallback_enabled else None
```

#### Componentes Principais do RAG

1. **Extração de Embeddings**:
   ```python
   def _encode_documents(self, documents):
       """
       Codifica documentos em embeddings.
       """
       try:
           return self.embedding_model.encode(
               documents, 
               convert_to_tensor=True, 
               normalize_embeddings=True
           )
       except RuntimeError as e:
           if "out of memory" in str(e) and self.fallback_enabled:
               return self._fallback_encode_documents(documents)
           raise
   ```

2. **Mecanismo de Recuperação**:
   ```python
   def retrieve(self, query, k=None):
       """
       Recupera documentos relevantes para a consulta.
       
       Args:
           query: Consulta para recuperação
           k: Número de documentos a recuperar (usa configuração padrão se None)
           
       Returns:
           Lista de tuplas (documento, score)
       """
       k = k or self.k_retrieval
       
       # Gerar embedding para consulta
       query_embedding = self._encode_query(query)
       
       # Recuperar documentos similares
       docs_with_scores = self.document_store.search(
           query_embedding, k=k, threshold=self.threshold
       )
       
       return docs_with_scores
   ```

3. **Integração com Modelo Principal**:
   ```python
   def augment_context(self, query, context=None):
       """
       Aumenta o contexto com informações relevantes recuperadas.
       
       Args:
           query: Consulta ou pergunta atual
           context: Contexto atual (opcional)
           
       Returns:
           Contexto aumentado com informações relevantes
       """
       retrieved_docs = self.retrieve(query)
       
       if not retrieved_docs:
           return context
           
       # Organizar informações recuperadas
       retrieved_info = []
       for doc, score in retrieved_docs:
           if score >= self.threshold:
               retrieved_info.append(f"{doc.content}")
               
       # Limitar tamanho do contexto aumentado
       augmented_text = " ".join(retrieved_info)
       if len(augmented_text) > self.max_context_length:
           augmented_text = augmented_text[:self.max_context_length] + "..."
           
       # Formatar para inclusão no contexto
       formatted_info = f"\nContexto adicional:\n{augmented_text}\n"
       
       # Combinar com contexto existente ou criar novo
       if context:
           augmented_context = context + formatted_info
       else:
           augmented_context = formatted_info
           
       return augmented_context
   ```

#### Mecanismo de Fallback para Hardware Limitado

```python
def _initialize_fallback(self):
    """
    Inicializa mecanismo de fallback para hardware limitado.
    
    Returns:
        Índice de fallback baseado em TF-IDF
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words=self.config.rag.get("stop_words", "portuguese")
    )
    
    return {
        "vectorizer": vectorizer,
        "initialized": False,
        "vectors": None,
        "documents": []
    }
    
def _fallback_encode_documents(self, documents):
    """
    Método alternativo de codificação para hardware limitado.
    
    Args:
        documents: Lista de documentos para codificar
        
    Returns:
        Representação TF-IDF dos documentos
    """
    vectorizer = self.fallback_index["vectorizer"]
    
    # Inicializar ou atualizar vetores
    if not self.fallback_index["initialized"]:
        vectors = vectorizer.fit_transform(documents)
        self.fallback_index["initialized"] = True
    else:
        vectors = vectorizer.transform(documents)
        
    self.fallback_index["documents"].extend(documents)
    
    return vectors
```

### Proatividade Contextual

O sistema de Proatividade Contextual permite que o LunaGPT ofereça sugestões e informações proativas baseadas no contexto da conversa:

```python
class ProactiveMessenger:
    """
    Sistema para geração de mensagens e sugestões proativas.
    
    Analisa o contexto conversacional para identificar oportunidades
    de fornecer informações úteis antes mesmo de serem solicitadas.
    """
    
    def __init__(self, config, luna_model):
        self.config = config
        self.model = luna_model
        self.enabled = config.proactive.get("enabled", False)
        self.threshold = config.proactive.get("threshold", 0.7)
        self.max_suggestions = config.proactive.get("max_suggestions", 2)
        
        # Padrões de oportunidade proativa
        self.opportunity_types = [
            "clarification",   # Esclarecer conceitos mencionados
            "extension",       # Expandir informações sobre tópico atual
            "related_concept", # Mencionar conceitos relacionados relevantes
            "practical_use",   # Sugerir aplicações práticas
            "contradiction"    # Apontar possíveis inconsistências
        ]
```

#### Detecção de Oportunidades Proativas

```python
def detect_opportunities(self, conversation_history):
    """
    Detecta oportunidades para mensagens proativas.
    
    Args:
        conversation_history: Histórico da conversa atual
        
    Returns:
        Lista de oportunidades detectadas com scores
    """
    if not self.enabled:
        return []
        
    # Extrair mensagens recentes  
    recent_messages = self._extract_recent_messages(conversation_history)
    if not recent_messages:
        return []
        
    # Preparar prompt para detector de oportunidades
    prompt = self._prepare_opportunity_prompt(recent_messages)
    
    # Gerar análise de oportunidades
    analysis = self._generate_opportunity_analysis(prompt)
    
    # Extrair oportunidades da análise
    opportunities = self._parse_opportunity_analysis(analysis)
    
    # Filtrar por threshold e limitar quantidade
    filtered_opportunities = [
        op for op in opportunities 
        if op["score"] >= self.threshold
    ]
    
    return filtered_opportunities[:self.max_suggestions]
```

#### Geração de Sugestões Proativas

```python
def generate_proactive_message(self, opportunity, conversation_history):
    """
    Gera uma mensagem proativa para uma oportunidade detectada.
    
    Args:
        opportunity: Dicionário com tipo e detalhes da oportunidade
        conversation_history: Histórico da conversa
        
    Returns:
        Mensagem proativa gerada
    """
    # Preparar prompt para geração proativa
    prompt = self._prepare_generation_prompt(opportunity, conversation_history)
    
    # Gerar mensagem
    message = self.model.generate_text(
        prompt, 
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        prefix="<|proactive|>"
    )
    
    # Formatar e validar mensagem
    message = self._format_proactive_message(message, opportunity["type"])
    
    return {
        "type": opportunity["type"],
        "content": message,
        "confidence": opportunity["score"]
    }
```

#### Integração com Fluxo Principal

```python
def enhance_response(self, response, conversation_history):
    """
    Aprimora resposta principal com elementos proativos.
    
    Args:
        response: Resposta principal gerada pelo modelo
        conversation_history: Histórico da conversa
        
    Returns:
        Resposta aprimorada com elementos proativos
    """
    if not self.enabled:
        return response
        
    # Detectar oportunidades
    opportunities = self.detect_opportunities(conversation_history)
    
    if not opportunities:
        return response
        
    # Gerar sugestões proativas
    proactive_messages = []
    for opportunity in opportunities:
        message = self.generate_proactive_message(opportunity, conversation_history)
        proactive_messages.append(message)
        
    # Integrar com resposta principal
    enhanced_response = self._integrate_proactive_messages(response, proactive_messages)
    
    return enhanced_response
```

Esta funcionalidade permite que o LunaGPT vá além de apenas responder perguntas, antecipando necessidades e fornecendo informações complementares valiosas.

### Curriculum Learning e Treinamento Progressivo

O LunaGPT implementa uma estratégia avançada de Curriculum Learning, que apresenta dados de treinamento em ordem crescente de complexidade:

```python
class CurriculumTrainer:
    """
    Implementação de Curriculum Learning para treinamento progressivo.
    
    Organiza o treinamento em estágios de dificuldade crescente, permitindo
    que o modelo aprenda conceitos fundamentais antes de enfrentar exemplos
    mais complexos.
    """
    
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # Configurações do curriculum
        self.stages = config.curriculum.get("stages", 3)
        self.metrics = config.curriculum.get("progress_metrics", ["loss", "accuracy"])
        self.progression_threshold = config.curriculum.get("progression_threshold", 0.8)
        self.patience = config.curriculum.get("patience", 2)
        
        # Estado do curriculum
        self.current_stage = 0
        self.stage_epochs = 0
        self.stage_progress = {metric: [] for metric in self.metrics}
```

#### Organização de Dados por Complexidade

```python
def prepare_curriculum_datasets(self, raw_datasets):
    """
    Organiza datasets em estágios de complexidade crescente.
    
    Args:
        raw_datasets: Conjuntos de dados brutos
        
    Returns:
        Lista de datasets organizados por complexidade
    """
    curriculum_datasets = []
    
    # Estágio 1: Exemplos curtos e simples
    simple_dataset = self._filter_by_complexity(
        raw_datasets, 
        max_length=50, 
        max_reasoning_depth=1
    )
    curriculum_datasets.append(simple_dataset)
    
    # Estágio 2: Exemplos de complexidade média
    medium_dataset = self._filter_by_complexity(
        raw_datasets, 
        max_length=150, 
        max_reasoning_depth=2
    )
    curriculum_datasets.append(medium_dataset)
    
    # Estágio 3: Conjunto completo
    curriculum_datasets.append(raw_datasets)
    
    return curriculum_datasets
```

#### Métricas de Progressão

```python
def evaluate_progression(self, eval_metrics):
    """
    Avalia se o modelo está pronto para progredir para o próximo estágio.
    
    Args:
        eval_metrics: Métricas da avaliação atual
        
    Returns:
        Boolean indicando se deve progredir para o próximo estágio
    """
    # Registrar métricas atuais
    for metric in self.metrics:
        if metric in eval_metrics:
            self.stage_progress[metric].append(eval_metrics[metric])
            
    self.stage_epochs += 1
    
    # Verificar se alcançou número mínimo de épocas
    min_epochs = self.config.curriculum.get("min_epochs_per_stage", 1)
    if self.stage_epochs < min_epochs:
        return False
        
    # Verificar se métricas estão acima do threshold
    ready_to_progress = True
    for metric in self.metrics:
        if len(self.stage_progress[metric]) < self.patience:
            return False
            
        # Verificar últimas N épocas (patience)
        recent_values = self.stage_progress[metric][-self.patience:]
        metric_threshold = self.progression_threshold
        
        if metric == "loss":  # Para loss, queremos valores abaixo do threshold
            if any(val > metric_threshold for val in recent_values):
                ready_to_progress = False
        else:  # Para outras métricas (accuracy, etc), queremos valores acima
            if any(val < metric_threshold for val in recent_values):
                ready_to_progress = False
                
    return ready_to_progress
```

#### Aplicação do Curriculum

```python
def train_with_curriculum(self, train_datasets, eval_dataset):
    """
    Executa treinamento com curriculum learning.
    
    Args:
        train_datasets: Lista de datasets organizados por complexidade
        eval_dataset: Dataset de validação
        
    Returns:
        Modelo treinado e histórico de métricas
    """
    training_history = []
    
    # Inicialização para primeiro estágio
    self.current_stage = 0
    self.stage_epochs = 0
    self.stage_progress = {metric: [] for metric in self.metrics}
    
    # Training loop para todos os estágios
    while self.current_stage < self.stages:
        print(f"Iniciando estágio {self.current_stage + 1}/{self.stages} do curriculum")
        
        # Configurar dataloader para estágio atual
        train_dataloader = self._prepare_dataloader(train_datasets[self.current_stage])
        
        # Treinar no estágio atual
        stage_metrics = self._train_stage(train_dataloader, eval_dataset)
        training_history.append({
            "stage": self.current_stage + 1,
            "epochs": self.stage_epochs,
            "metrics": stage_metrics
        })
        
        # Verificar progressão para próximo estágio
        should_progress = self.evaluate_progression(stage_metrics[-1])
        
        if should_progress and self.current_stage < self.stages - 1:
            self.current_stage += 1
            self.stage_epochs = 0
            self.stage_progress = {metric: [] for metric in self.metrics}
            
            # Aplicar técnicas de transição entre estágios
            self._apply_stage_transition()
        elif not should_progress and self.current_stage == self.stages - 1:
            # Finalizou último estágio com sucesso
            break
            
    return self.model, training_history
```

### Sistema de Feedback e Refinamento Contínuo

O LunaGPT implementa um sofisticado sistema de feedback e refinamento que permite ao modelo melhorar continuamente com base nas interações com usuários:

```python
class FeedbackSystem:
    """
    Sistema de coleta e processamento de feedback para refinamento contínuo.
    
    Permite capturar feedback multidimensional dos usuários e utilizá-lo para
    ajustes contínuos no comportamento do modelo.
    """
    
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path
        self.feedback_db_path = config.feedback.get("db_path", "feedback.jsonl")
        self.feedback_dimensions = config.feedback.get("dimensions", [
            "helpful", "relevant", "accurate", "detailed", "concise"
        ])
        self.reflection_enabled = config.feedback.get("enable_reflection", True)
```

#### Coleta de Feedback Multidimensional

```python
def collect_feedback(self, conversation_id, response_id, feedback_data):
    """
    Coleta feedback do usuário sobre uma resposta específica.
    
    Args:
        conversation_id: ID da conversa
        response_id: ID da resposta específica
        feedback_data: Dict com feedback estruturado
    
    Returns:
        ID do feedback registrado
    """
    # Validar feedback
    valid_dimensions = set(self.feedback_dimensions + ["text_feedback", "corrections"])
    for key in feedback_data:
        if key not in valid_dimensions:
            raise ValueError(f"Dimensão de feedback inválida: {key}")
            
    # Adicionar metadados
    feedback_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "response_id": response_id,
        "feedback": feedback_data
    }
    
    # Armazenar feedback
    with open(self.feedback_db_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(feedback_entry) + '\n')
        
    # Gerar reflexão se habilitado
    if self.reflection_enabled and feedback_data.get("text_feedback"):
        self._generate_reflection(feedback_entry)
        
    return feedback_entry["id"]
```

#### Análise e Integração de Feedback

```python
def analyze_feedback(self, time_window=None, min_entries=50):
    """
    Analisa feedback coletado para identificar padrões.
    
    Args:
        time_window: Janela temporal para análise (None para todos)
        min_entries: Mínimo de entradas para análise significativa
        
    Returns:
        Relatório de análise de feedback
    """
    # Carregar feedback
    feedback_entries = self._load_feedback(time_window)
    
    if len(feedback_entries) < min_entries:
        return {"status": "insufficient_data", "count": len(feedback_entries)}
        
    # Analisar por dimensão
    dimension_stats = {}
    for dim in self.feedback_dimensions:
        scores = [e["feedback"].get(dim) for e in feedback_entries if dim in e["feedback"]]
        if scores:
            dimension_stats[dim] = {
                "mean": sum(scores) / len(scores),
                "median": statistics.median(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "count": len(scores)
            }
            
    # Identificar pontos fracos (dimensões com pontuações mais baixas)
    weakness_threshold = self.config.feedback.get("weakness_threshold", 3.5)
    weaknesses = [dim for dim, stats in dimension_stats.items() 
                  if stats["mean"] < weakness_threshold]
    
    # Extrair feedback textual mais relevante
    text_feedback = [e["feedback"].get("text_feedback") for e in feedback_entries 
                    if "text_feedback" in e["feedback"] and e["feedback"]["text_feedback"]]
    
    # Analisar padrões em feedback textual (análise simplificada)
    text_analysis = self._analyze_text_feedback(text_feedback) if text_feedback else {}
    
    return {
        "status": "success",
        "count": len(feedback_entries),
        "dimension_stats": dimension_stats,
        "identified_weaknesses": weaknesses,
        "common_themes": text_analysis.get("common_themes", []),
        "top_keywords": text_analysis.get("top_keywords", [])
    }
```

#### Geração de Reflexões e Auto-Refinamento

```python
def _generate_reflection(self, feedback_entry):
    """
    Gera reflexão sobre o feedback para auto-aprendizado.
    
    Args:
        feedback_entry: Entrada de feedback completa
        
    Returns:
        ID da reflexão gerada
    """
    from .luna_model import LunaModel
    
    # Carregar modelo em modo leve
    model = LunaModel.from_pretrained(
        self.model_path, 
        use_lightweight=True
    )
    
    # Construir prompt de reflexão
    prompt = f"""
    Analise criticamente o feedback a seguir sobre uma resposta sua:
    
    Resposta original: {feedback_entry.get('original_response', '[Resposta indisponível]')}
    
    Feedback do usuário:
    """
    
    feedback_data = feedback_entry["feedback"]
    for dim in self.feedback_dimensions:
        if dim in feedback_data:
            score = feedback_data[dim]
            prompt += f"\n- {dim.capitalize()}: {score}/5"
    
    if "text_feedback" in feedback_data:
        prompt += f"\n\nComentário textual: {feedback_data['text_feedback']}"
    
    if "corrections" in feedback_data:
        prompt += f"\n\nCorreções sugeridas: {feedback_data['corrections']}"
    
    prompt += """
    
    Com base neste feedback, realize uma reflexão estruturada:
    1. Quais pontos fortes foram identificados na resposta?
    2. Quais áreas precisam de melhoria?
    3. Como você abordaria diferentemente uma pergunta similar no futuro?
    4. Que estratégias específicas poderia implementar para melhorar?
    """
    
    # Gerar reflexão
    reflection = model.generate_text(prompt, max_length=500, temperature=0.7)
    
    # Armazenar reflexão
    reflection_id = str(uuid.uuid4())
    reflection_entry = {
        "id": reflection_id,
        "feedback_id": feedback_entry["id"],
        "timestamp": datetime.datetime.now().isoformat(),
        "reflection": reflection
    }
    
    # Salvar na base de reflexões
    reflection_path = os.path.join(
        os.path.dirname(self.feedback_db_path), 
        "reflections.jsonl"
    )
    with open(reflection_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(reflection_entry) + '\n')
    
    return reflection_id
```

### Personas e Adaptabilidade de Resposta

O LunaGPT implementa um sistema avançado de personas que permite adaptar o estilo, tom e abordagem das respostas:

```python
class PersonaManager:
    """
    Sistema de gestão e aplicação de personas para adaptação contextual.
    
    Permite moldar dinamicamente as respostas em diferentes estilos,
    adequando o tom, complexidade e abordagem para diferentes contextos
    e necessidades dos usuários.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Carregar definições de personas
        self.personas_path = config.personas.get("definitions_path", "personas.json")
        self.personas = self._load_personas()
        
        # Persona padrão
        self.default_persona_id = config.personas.get("default", "balanced")
```

#### Definições de Personas

```python
def _load_personas(self):
    """
    Carrega definições de personas do arquivo de configuração.
    
    Returns:
        Dicionário de definições de personas
    """
    try:
        with open(self.personas_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback para personas padrão
        return {
            "balanced": {
                "name": "Equilibrado",
                "description": "Abordagem equilibrada com precisão e clareza",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                },
                "style": {
                    "formality": 0.5,  # 0=informal, 1=formal
                    "complexity": 0.5,  # 0=simples, 1=complexo
                    "verbosity": 0.5,   # 0=conciso, 1=detalhado
                    "creativity": 0.5   # 0=conservador, 1=criativo
                },
                "prompt_modifiers": [
                    "Forneça uma resposta equilibrada com precisão e clareza."
                ]
            },
            "academic": {
                "name": "Acadêmico",
                "description": "Estilo formal e rigoroso com ênfase em precisão",
                "parameters": {
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.0
                },
                "style": {
                    "formality": 0.9,
                    "complexity": 0.8,
                    "verbosity": 0.7,
                    "creativity": 0.3
                },
                "prompt_modifiers": [
                    "Responda em estilo acadêmico com precisão e rigor científico.",
                    "Cite fontes relevantes quando apropriado."
                ]
            },
            "friendly": {
                "name": "Amigável",
                "description": "Estilo conversacional e acessível",
                "parameters": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.2
                },
                "style": {
                    "formality": 0.2,
                    "complexity": 0.3,
                    "verbosity": 0.5,
                    "creativity": 0.7
                },
                "prompt_modifiers": [
                    "Responda de forma amigável e conversacional.",
                    "Use linguagem acessível e evite jargão desnecessário."
                ]
            }
        }
```

#### Aplicação de Personas

```python
def apply_persona(self, persona_id, generation_config):
    """
    Aplica configurações de uma persona ao processo de geração.
    
    Args:
        persona_id: Identificador da persona a ser aplicada
        generation_config: Configuração atual de geração
        
    Returns:
        Configuração de geração modificada pela persona
    """
    # Usar persona padrão se a solicitada não for encontrada
    if persona_id not in self.personas:
        persona_id = self.default_persona_id
        
    persona = self.personas[persona_id]
    
    # Aplicar parâmetros de geração
    for param, value in persona.get("parameters", {}).items():
        generation_config[param] = value
        
    # Adicionar modificadores de prompt se aplicável
    if "prompt_suffix" in generation_config and persona.get("prompt_modifiers"):
        modifiers = " ".join(persona["prompt_modifiers"])
        generation_config["prompt_suffix"] = f"{modifiers} {generation_config.get('prompt_suffix', '')}"
        
    return generation_config
```

#### Detecção Automática de Persona Ideal

```python
def detect_ideal_persona(self, user_message, conversation_history=None):
    """
    Detecta a persona mais adequada para responder à mensagem do usuário.
    
    Args:
        user_message: Mensagem atual do usuário
        conversation_history: Histórico da conversa (opcional)
        
    Returns:
        ID da persona mais adequada
    """
    # Análise básica de características da mensagem
    message_length = len(user_message.split())
    has_question_mark = "?" in user_message
    has_technical_terms = self._detect_technical_terms(user_message)
    formality_score = self._assess_formality(user_message)
    
    # Análise do histórico (se disponível)
    user_preference = None
    if conversation_history and len(conversation_history) >= 3:
        user_preference = self._analyze_user_preference(conversation_history)
        
    # Lógica de decisão para seleção de persona
    if has_technical_terms and formality_score > 0.7:
        return "academic"
    elif message_length < 15 and not has_technical_terms:
        return "friendly"
    elif user_preference:
        return user_preference
    else:
        return self.default_persona_id
```

Esta funcionalidade permite que o LunaGPT adapte sua comunicação a diferentes contextos e preferências dos usuários, resultando em interações mais naturais e eficazes.

---

## Instalação e Configuração

### Requisitos de Sistema Detalhados

O LunaGPT foi projetado para funcionar em uma ampla gama de sistemas, com requisitos adaptativos dependendo do modo de operação:

#### Requisitos Mínimos (Modo Leve)
- **Sistema Operacional**: Windows 10+, Ubuntu 18.04+, macOS 10.14+
- **CPU**: Processador quad-core, 2.5GHz+
- **RAM**: 8GB (16GB recomendado)
- **Espaço em Disco**: 5GB para o modelo básico
- **Python**: 3.8 ou superior

#### Requisitos Recomendados (Funcionalidades Completas)
- **Sistema Operacional**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **CPU**: Processador octa-core, 3.0GHz+
- **RAM**: 16GB (32GB para RAG completo)
- **GPU**: NVIDIA com 6GB+ VRAM (10GB+ recomendado)
- **Espaço em Disco**: 15GB
- **Python**: 3.9 ou superior
- **CUDA**: 11.7+ (para aceleração NVIDIA)
- **Conexão Internet**: Recomendado para RAG e atualizações de modelo

#### Bibliotecas e Dependências

**Dependências Principais**:
```
torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.3
numpy>=1.22.0
scipy>=1.8.0
tqdm>=4.64.0
```

**Dependências Opcionais (para todas as funcionalidades)**:
```
sentence-transformers>=2.2.2  # Para RAG
faiss-cpu>=1.7.0             # Para busca vetorial em CPU
faiss-gpu>=1.7.0             # Para busca vetorial em GPU
scikit-learn>=1.0.0          # Para fallbacks e avaliação
nltk>=3.7                    # Para processamento de texto
matplotlib>=3.5.0            # Para visualizações
wandb>=0.13.0                # Para rastreamento de experimentos
```

### Instalação Passo-a-Passo

#### 1. Configuração do Ambiente

**Usando Conda (Recomendado)**:
```bash
# Criar ambiente
conda create -n lunagpt python=3.9
conda activate lunagpt

# Instalar PyTorch com suporte CUDA (verificar compatibilidade em pytorch.org)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

**Usando venv**:
```bash
# Criar ambiente
python -m venv lunagpt-env
# Ativar ambiente (Windows)
lunagpt-env\Scripts\activate
# Ativar ambiente (Linux/macOS)
source lunagpt-env/bin/activate

# Instalar PyTorch com suporte CUDA (verificar em pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### 2. Instalação do LunaGPT

**Instalação a partir do repositório**:
```bash
# Clonar repositório
git clone https://github.com/seu-usuario/lunagpt.git
cd lunagpt

# Instalação básica
pip install -e .

# Instalação com todas as funcionalidades
pip install -e .[all]

# Instalação específica para GPU
pip install -e .[gpu]

# Instalação para desenvolvimento
pip install -e .[dev]
```

**Instalação via pip (quando disponível)**:
```bash
# Instalação básica
pip install lunagpt

# Instalação completa
pip install lunagpt[all]
```

### Verificação de Instalação

Para verificar se a instalação do LunaGPT foi concluída com sucesso, execute o script de verificação integrado:

```bash
# A partir do diretório do LunaGPT
python -m lunagpt.utils.verify_installation

# Teste de GPU (se aplicável)
python -m lunagpt.utils.verify_gpu
```
O script de verificação realizará as seguintes checagens:

1. **Versão do Python**: Verifica se a versão do Python é compatível (3.8+)
2. **Dependências instaladas**: Confirma que todas as bibliotecas necessárias estão instaladas corretamente
3. **Disponibilidade de GPU**: Detecta se há GPU disponível e se o PyTorch está configurado para usá-la
4. **Modelos pré-carregados**: Verifica se os modelos base podem ser carregados
5. **Funcionamento básico**: Executa um teste simples de geração de texto
6. **Compatibilidade do tokenizador**: Valida se o tokenizador está funcionando corretamente
7. **Sistema RAG**: Testa componentes do sistema de Retrieval-Augmented Generation (quando habilitado)
8. **Permissões**: Confirma permissões de leitura/escrita nos diretórios necessários

O resultado da verificação será exibido com informações detalhadas sobre cada teste, incluindo possíveis soluções para problemas encontrados.

### Configuração Avançada

#### Configuração do Sistema

O LunaGPT usa um sistema de configuração baseado em arquivos JSON ou YAML. O arquivo de configuração padrão pode ser encontrado em `configs/default.json`, mas você pode criar suas próprias configurações personalizadas.

**Exemplo de arquivo de configuração básico:**
```json
{
  "model": {
    "name": "luna-2.6",
    "base_path": "./models/luna-2.6",
    "use_moe": true,
    "use_state_space": true,
    "use_hypernet": false,
    "use_growing": false
  },
  "tokenizer": {
    "vocab_path": "./models/luna-2.6/tokenizer/vocab.json",
    "merges_path": "./models/luna-2.6/tokenizer/merges.txt",
    "max_length": 2048,
    "truncation_strategy": "right"
  },
  "hardware": {
    "auto_detect": true,
    "force_cpu": false,
    "mixed_precision": true,
    "quantization": "int8",
    "gpu_memory_fraction": 0.9
  },
  "rag": {
    "enabled": true,
    "k_retrieval": 5,
    "relevance_threshold": 0.7,
    "enable_fallback": true
  },
  "proactive": {
    "enabled": true,
    "threshold": 0.7,
    "max_suggestions": 2
  },
  "personas": {
    "default": "balanced",
    "definitions_path": "./configs/personas.json"
  }
}
```

**Para usar uma configuração personalizada:**

```bash
# Por linha de comando
python -m lunagpt.cli --config path/to/custom_config.json

# Programaticamente
from lunagpt.config import LunaConfig

config = LunaConfig.from_file("path/to/custom_config.json")
model = LunaModel.from_config(config)
```

#### Customização de Hardware

O LunaGPT oferece várias opções para otimização de hardware:

```json
{
  "hardware": {
    "auto_detect": true,             // Detectar recursos de hardware automaticamente
    "force_cpu": false,              // Forçar execução em CPU mesmo com GPU disponível
    "gpu_devices": [0, 1],           // Índices de GPUs a utilizar (multi-GPU)
    "mixed_precision": true,         // Usar precisão mista (FP16) quando possível
    "quantization": "int8",          // Tipo de quantização (int8, int4 ou null)
    "gpu_memory_fraction": 0.9,      // Fração da memória GPU a utilizar
    "cpu_threads": 4,                // Número de threads para operações em CPU
    "offload_strategy": "balanced"   // Estratégia de offload para GPU limitada (none, balanced, aggressive)
  }
}
```

Para hardware limitado, configurações específicas estão disponíveis:

```json
{
  "hardware": {
    "low_memory": true,              // Ativa otimizações para baixa memória
    "sequential_offload": true,      // Carrega camadas sequencialmente para reduzir pico de memória
    "cache_limit_mb": 512,           // Limita cache de atenção
    "quantization": "int4",          // Quantização mais agressiva
    "skip_components": ["moe"]       // Desativa componentes intensivos em memória
  }
}
```

#### Configuração da Interface de Linha de Comando

A CLI (Command Line Interface) pode ser customizada através de argumentos de linha de comando:

```bash
# Iniciar em modo interativo com configuração personalizada
python -m lunagpt.cli --interactive --config custom_config.json

# Processar arquivo de texto
python -m lunagpt.cli --input input.txt --output output.txt

# Usar persona específica
python -m lunagpt.cli --interactive --persona academic

# Ativar modo verbose para depuração
python -m lunagpt.cli --interactive --verbose
```

### Soluções para Problemas Comuns de Instalação

#### Problemas com CUDA

**Sintoma**: Erros relacionados a CUDA mesmo com GPU disponível.

**Soluções**:
1. Verificar compatibilidade entre versão do PyTorch e versão do CUDA:
   ```bash
   # Verificar versão do PyTorch e suporte CUDA
   python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA disponível: {torch.cuda.is_available()}')"
   
   # Ver dispositivos CUDA
   python -c "import torch; print(f'Dispositivos: {torch.cuda.device_count()}'); [print(f'- {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
   ```

2. Reinstalar PyTorch com a versão CUDA correta:
   ```bash
   # Desinstalar PyTorch atual
   pip uninstall torch torchvision torchaudio
   
   # Reinstalar com versão CUDA compatível
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```

#### Erros ao Carregar Modelos

**Sintoma**: Falha ao carregar o modelo com erros de chave ou formato.

**Soluções**:
1. Verificar integridade dos arquivos do modelo:
   ```bash
   python -m lunagpt.utils.verify_model_files --model-path ./models/luna-2.6
   ```

2. Baixar novamente os arquivos do modelo:
   ```bash
   python -m lunagpt.utils.download_model --model luna-2.6 --force
   ```

#### Problemas de Memória

**Sintoma**: Erros de falta de memória (OOM - Out Of Memory).

**Soluções**:
1. Ativar otimizações para baixa memória no arquivo de configuração:
   ```json
   {
     "hardware": {
       "low_memory": true,
       "sequential_offload": true,
       "quantization": "int8"
     }
   }
   ```

2. Reduzir tamanho do contexto:
   ```json
   {
     "tokenizer": {
       "max_length": 512  // Valor menor que o padrão
     }
   }
   ```

3. Desativar componentes avançados:
   ```bash
   python -m lunagpt.cli --disable-moe --disable-rag
   ```

#### Problemas com o Tokenizador

**Sintoma**: Erros ao carregar o tokenizador ou tokens incorretos.

**Soluções**:
1. Verificar arquivos do tokenizador:
   ```bash
   python -m lunagpt.utils.verify_tokenizer --tokenizer-path ./models/luna-2.6/tokenizer
   ```

2. Reconstruir cache do tokenizador:
   ```bash
   python -m lunagpt.utils.rebuild_tokenizer_cache
   ```

---

## Guia de Utilização

### Primeiros Passos (Para Iniciantes)

Para começar a usar o LunaGPT, siga estas etapas básicas:

#### 1. Executar o Sistema em Modo Interativo

O modo interativo é a forma mais simples de começar a usar o LunaGPT:

```bash
# Iniciar o LunaGPT em modo interativo
python -m lunagpt.cli --interactive
```

Você verá uma interface de linha de comando onde pode digitar mensagens e receber respostas do modelo:

```
=== LunaGPT v2.6 ===
Digite sua mensagem (ou 'sair' para encerrar):

Você: Olá, como você funciona?

LunaGPT: Olá! Eu sou o LunaGPT, um sistema de diálogo neural baseado em uma arquitetura híbrida que combina transformers, state-space models e mixture of experts. Funciono processando seu texto através de várias camadas neurais especializadas que foram treinadas em grandes volumes de texto em português.

Quando você me envia uma mensagem, eu a tokenizo (transformo em unidades que posso processar), analiso seu contexto, e gero uma resposta utilizando meus parâmetros internos. Se ativado, também posso buscar informações adicionais em uma base de conhecimento para enriquecer minhas respostas.

Em que mais posso ajudar você hoje?

Você: 
```

#### 2. Experimentar Personas Diferentes

Você pode testar diferentes estilos de resposta usando o parâmetro `--persona`:

```bash
# Usar persona acadêmica
python -m lunagpt.cli --interactive --persona academic

# Usar persona amigável
python -m lunagpt.cli --interactive --persona friendly

# Alternar personas durante a sessão
# (Digite '/persona nome_da_persona' durante a conversa)
```

#### 3. Usar o Sistema RAG

Para usar o sistema RAG com documentos próprios:

```bash
# Indexar documentos para o sistema RAG
python -m lunagpt.utils.index_documents --dir ./meus_documentos --output ./index_rag

# Iniciar com base de conhecimento personalizada
python -m lunagpt.cli --interactive --rag --knowledge-base ./index_rag
```

#### 4. Experimentar o Modo de Pensamento Visível

O modo de "pensamento visível" (thinking) mostra o raciocínio interno do modelo:

```bash
# Ativar modo thinking
python -m lunagpt.cli --interactive --show-thinking
```

Exemplo de saída:

```
Você: Quais são os aspectos mais importantes ao desenvolver um sistema de IA?

LunaGPT (pensando): Vou abordar esta questão considerando múltiplas dimensões do desenvolvimento de IA:
1. Aspectos técnicos: arquitetura, dados, avaliação
2. Aspectos éticos: viés, transparência, privacidade
3. Aspectos práticos: implementação, escalabilidade
Vou organizar uma resposta que cubra estas dimensões de forma equilibrada, começando com fundamentos técnicos, passando por considerações éticas e terminando com aspectos práticos.

LunaGPT: O desenvolvimento de sistemas de IA eficazes e responsáveis envolve diversos aspectos fundamentais:

**Aspectos Técnicos:**
- Qualidade e representatividade dos dados de treinamento
- Arquitetura adequada à tarefa e domínio específicos
- Métodos robustos de avaliação e validação
- Capacidade de explicabilidade das decisões do modelo

**Considerações Éticas:**
- Mitigação de vieses e discriminação algorítmica
- Transparência no funcionamento e limitações
- Respeito à privacidade e proteção de dados
- Mecanismos de supervisão humana apropriados

**Implementação Prática:**
- Escalabilidade e eficiência computacional
- Facilidade de integração com sistemas existentes
- Monitoramento contínuo de desempenho
- Atualização e manutenção contínuas

O equilíbrio entre estes aspectos é essencial para criar sistemas que não apenas funcionem bem tecnicamente, mas também sejam confiáveis, justos e valiosos para os usuários.

Você:
```

### Interface de Linha de Comando

O LunaGPT oferece uma CLI completa com diversas opções para diferentes casos de uso:

#### Comandos Básicos

```bash
# Modo interativo (chat)
python -m lunagpt.cli --interactive

# Processamento em lote (arquivos)
python -m lunagpt.cli --input perguntas.txt --output respostas.txt

# Processamento de consulta única
python -m lunagpt.cli --query "Quais são as principais características do LunaGPT?"
```

#### Opções de Configuração

```bash
# Usar arquivo de configuração personalizado
python -m lunagpt.cli --config minha_config.json

# Definir temperatura (aleatoriedade)
python -m lunagpt.cli --interactive --temperature 0.8

# Definir comprimento máximo de resposta
python -m lunagpt.cli --interactive --max-tokens 300

# Definir contexto do sistema
python -m lunagpt.cli --interactive --system "Você é um assistente especializado em finanças"
```

#### Comandos Especiais Durante o Chat

Durante uma sessão interativa, você pode usar comandos especiais prefixados com `/`:

```
/help              # Mostrar ajuda de comandos
/clear             # Limpar histórico da conversa
/save [arquivo]    # Salvar conversa atual
/load [arquivo]    # Carregar conversa de arquivo
/persona [nome]    # Mudar para persona específica
/rag on/off        # Ativar/desativar sistema RAG
/thinking on/off   # Mostrar/ocultar pensamento interno
/reset             # Reiniciar o modelo
/exit ou /quit     # Sair da aplicação
```

#### Modo de Lote com Arquivo JSON

Para processar múltiplas consultas com configurações individuais:

```json
// consultas.json
[
  {
    "query": "O que é aprendizado profundo?",
    "persona": "academic",
    "max_tokens": 250
  },
  {
    "query": "Me dê uma receita de pão caseiro",
    "persona": "friendly",
    "temperature": 0.8
  }
]
```

```bash
# Processar arquivo JSON
python -m lunagpt.cli --batch consultas.json --output respostas.json
```

### API Python para Integração Programática

Para integrar o LunaGPT em seus próprios aplicativos Python, utilize a API do modelo:

```python
from lunagpt import LunaModel, LunaConfig
from lunagpt.chat import LunaChat

# Carregar configuração
config = LunaConfig.from_file("config.json")

# Inicializar modelo
model = LunaModel.from_pretrained("./models/luna-2.6", config=config)

# Criar instância de chat
chat = LunaChat(model=model)

# Exemplo de conversa simples
response = chat.send_message("Como funciona o sistema RAG no LunaGPT?")
print(response)

# Exemplo com histórico de conversa
messages = [
    {"role": "system", "content": "Você é um assistente especializado em IA."},
    {"role": "user", "content": "O que é aprendizado profundo?"},
    {"role": "assistant", "content": "Aprendizado profundo é uma subárea de machine learning..."},
    {"role": "user", "content": "Como isso se relaciona com redes neurais?"}
]
response = chat.send_message_with_history("Como isso se relaciona com redes neurais?", messages)
print(response)

# Uso com persona específica
response = chat.send_message(
    "Explique o que é um transformador em IA",
    persona="academic"
)
print(response)

# Uso com RAG
chat.enable_rag(knowledge_base_path="./conhecimento")
response = chat.send_message("Quais são as características do LunaGPT?")
print(response)
```

#### Exemplo de Servidor Web Simples

```python
from flask import Flask, request, jsonify
from lunagpt import LunaModel
from lunagpt.chat import LunaChat

app = Flask(__name__)

# Inicializar modelo e chat (em produção, faça isso fora da definição da aplicação)
model = LunaModel.from_pretrained("./models/luna-2.6")
chat_instances = {}

@app.route('/chat/<session_id>', methods=['POST'])
def chat_endpoint(session_id):
    # Obter ou criar instância de chat para a sessão
    if session_id not in chat_instances:
        chat_instances[session_id] = LunaChat(model=model)
    
    # Obter mensagem da requisição
    data = request.json
    message = data.get('message', '')
    persona = data.get('persona', 'balanced')
    
    # Gerar resposta
    response = chat_instances[session_id].send_message(
        message, persona=persona
    )
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Fluxos de Trabalho Comuns

#### 1. Assistente de Pesquisa com RAG

```python
from lunagpt import LunaModel, LunaConfig
from lunagpt.rag import RAGSystem
from lunagpt.chat import LunaChat

# Configurar sistema
config = LunaConfig.from_file("config.json")
model = LunaModel.from_pretrained("./models/luna-2.6", config=config)

# Indexar documentos de pesquisa
rag = RAGSystem(config=config)
rag.index_directory("./papers/")

# Criar chat com RAG
chat = LunaChat(model=model, rag_system=rag)

# Definir contexto de sistema
system_prompt = """
Você é um assistente de pesquisa especializado em analisar artigos científicos.
Responda com base apenas nos documentos fornecidos pelo sistema RAG.
Cite as fontes específicas para suas afirmações.
"""
chat.set_system_prompt(system_prompt)

# Interagir com o assistente
response = chat.send_message(
    "Quais são as metodologias mais recentes mencionadas nesses artigos?"
)
print(response)
```

#### 2. Geração de Conteúdo com Diferentes Estilos

```python
from lunagpt import LunaModel
from lunagpt.personas import PersonaManager

# Inicializar modelo
model = LunaModel.from_pretrained("./models/luna-2.6")

# Inicializar gerenciador de personas
persona_manager = PersonaManager(config=model.config)

# Gerar conteúdo em diferentes estilos
topic = "Inteligência Artificial"

# Versão acadêmica
academic_config = persona_manager.apply_persona("academic", {})
academic_content = model.generate_text(
    f"Escreva sobre {topic}",
    **academic_config
)

# Versão informal
friendly_config = persona_manager.apply_persona("friendly", {})
friendly_content = model.generate_text(
    f"Escreva sobre {topic}",
    **friendly_config
)

# Versão técnica
technical_config = persona_manager.apply_persona("technical", {})
technical_content = model.generate_text(
    f"Escreva sobre {topic}",
    **technical_config
)

# Salvar os diferentes estilos
with open("conteudo_comparativo.txt", "w") as f:
    f.write(f"# Conteúdo Acadêmico\n\n{academic_content}\n\n")
    f.write(f"# Conteúdo Amigável\n\n{friendly_content}\n\n")
    f.write(f"# Conteúdo Técnico\n\n{technical_content}\n\n")
```

#### 3. Análise de Feedback e Refinamento

```python
from lunagpt import LunaModel
from lunagpt.feedback import FeedbackSystem

# Inicializar modelo
model = LunaModel.from_pretrained("./models/luna-2.6")

# Inicializar sistema de feedback
feedback_system = FeedbackSystem(config=model.config, model_path="./models/luna-2.6")

# Analisar feedback existente
feedback_analysis = feedback_system.analyze_feedback(time_window="last_month")

# Identificar áreas de melhoria
if feedback_analysis["status"] == "success":
    weaknesses = feedback_analysis["identified_weaknesses"]
    print(f"Áreas para melhoria: {weaknesses}")
    
    # Gerar estratégias de refinamento baseadas no feedback
    improvement_prompt = f"""
    Com base na análise de feedback do usuário, o modelo precisa melhorar
    nas seguintes áreas: {', '.join(weaknesses)}.
    
    Sugira estratégias específicas para refinar o modelo nestas dimensões.
    """
    
    improvement_strategies = model.generate_text(
        improvement_prompt,
        max_length=500,
        temperature=0.7
    )
    
    print("\nEstratégias de Refinamento:")
    print(improvement_strategies)
```

### Cenários de Uso Avançado

#### 1. Integração com Sistema de Busca Vetorial Próprio

```python
import numpy as np
import faiss
from lunagpt import LunaModel
from lunagpt.chat import LunaChat

# Inicializar modelo
model = LunaModel.from_pretrained("./models/luna-2.6")

# Criar índice FAISS personalizado
dimension = 768  # Dimensão dos embeddings
index = faiss.IndexFlatIP(dimension)  # Índice de produto interno (similaridade de cosseno)

# Classe de RAG personalizada
class CustomRAG:
    def __init__(self, model, index, documents):
        self.model = model
        self.index = index
        self.documents = documents
        
    def encode_query(self, query):
        # Gerar embedding para a consulta
        return self.model.encode_text(query, pooling="mean")
        
    def retrieve(self, query, k=5):
        # Codificar consulta
        query_vector = self.encode_query(query)
        
        # Normalizar vetor (para similaridade de cosseno)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Buscar no índice
        scores, indices = self.index.search(
            query_vector.reshape(1, -1).astype('float32'), k
        )
        
        # Retornar documentos encontrados
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and scores[0][i] > 0.7:  # Threshold de relevância
                results.append({
                    "content": self.documents[idx],
                    "score": float(scores[0][i])
                })
                
        return results
        
    def augment_context(self, query, context=None):
        # Recuperar documentos relevantes
        retrieved_docs = self.retrieve(query)
        
        if not retrieved_docs:
            return context or ""
            
        # Formatar documentos recuperados
        docs_text = "\n\n".join([
            f"[Documento {i+1} (Score: {doc['score']:.2f})]\n{doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Adicionar ao contexto
        augmented = f"Documentos relevantes:\n{docs_text}\n\n"
        if context:
            augmented += context
            
        return augmented

# Exemplo de uso
documents = [
    "O LunaGPT utiliza uma arquitetura híbrida com transformers e state-space models.",
    "Mixture of Experts permite eficiência computacional mantendo alta capacidade.",
    "O sistema RAG do LunaGPT recupera informações relevantes de uma base de conhecimento."
    # Adicionar mais documentos...
]

# Codificar documentos
document_vectors = np.array([
    model.encode_text(doc, pooling="mean") 
    for doc in documents
])

# Normalizar vetores
norms = np.linalg.norm(document_vectors, axis=1, keepdims=True)
document_vectors = document_vectors / norms

# Adicionar ao índice
index.add(document_vectors.astype('float32'))

# Criar RAG personalizado
custom_rag = CustomRAG(model, index, documents)

# Criar chat com RAG personalizado
chat = LunaChat(model=model)

# Usar em consulta
response = chat.send_message_with_custom_rag(
    "Como o LunaGPT utiliza Mixture of Experts?",
    custom_rag
)
print(response)
```

#### 2. Fine-tuning com Feedback Específico

```python
from lunagpt import LunaModel, LunaConfig
from lunagpt.training import Trainer
import torch
from torch.utils.data import Dataset, DataLoader

# Classe de dataset personalizada para feedback
class FeedbackDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []
        
        # Processar exemplos
        for item in data:
            # Tokenizar entrada original
            original_input = item["original_query"]
            original_response = item["original_response"]
            improved_response = item["improved_response"]
            
            # Criar exemplo de treinamento (formato: query + original + improved)
            example = f"Query: {original_input}\nOriginal: {original_response}\nMelhorado: {improved_response}"
            tokenized = tokenizer.encode(example, return_tensors="pt")
            
            self.examples.append({
                "input_ids": tokenized,
                "labels": tokenized  # Autoregressive training
            })
    
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

# Carregar modelo
config = LunaConfig.from_file("config_training.json")
model = LunaModel.from_pretrained("./models/luna-2.6", config=config)

# Preparar dados de feedback
feedback_data = [
    {
        "original_query": "O que é aprendizado de máquina?",
        "original_response": "Aprendizado de máquina é uma tecnologia onde computadores aprendem.",
        "improved_response": "Aprendizado de máquina é um subcampo da inteligência artificial que se concentra no desenvolvimento de algoritmos e modelos estatísticos que permitem que sistemas computacionais 'aprendam' com dados, identificando padrões e tomando decisões com mínima intervenção humana. Diferentemente da programação tradicional, onde regras explícitas são codificadas, no aprendizado de máquina o sistema desenvolve suas próprias regras a partir de exemplos."
    },
    # Mais exemplos...
]

# Criar dataset
dataset = FeedbackDataset(feedback_data, model.tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Configurar treinador
trainer = Trainer(
    model=model,
    train_dataloader=dataloader,
    learning_rate=5e-5,
    num_epochs=3,
    gradient_accumulation_steps=4,
    checkpoint_dir="./checkpoints/feedback_tuning"
)

# Executar fine-tuning
trainer.train()

# Salvar modelo refinado
model.save("./models/luna-2.6-refined")

# Testar modelo refinado
test_query = "Explique o que é redes neurais."
response = model.generate_text(test_query)
print(response)
```

#### 3. Sistema Multi-Modal (Texto + Imagens)

Para cenários avançados que exigem integração com elementos multimodais:

```python
import torch
from PIL import Image
from lunagpt import LunaModel
from transformers import AutoFeatureExtractor, AutoModel

# Carregar Luna para processamento de texto
luna_model = LunaModel.from_pretrained("./models/luna-2.6")

# Carregar modelo de visão para processar imagens
vision_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vision_model = AutoModel.from_pretrained("google/vit-base-patch16-224")

class MultiModalLuna:
    def __init__(self, luna_model, vision_extractor, vision_model):
        self.luna = luna_model
        self.vision_extractor = vision_extractor
        self.vision_model = vision_model
        self.vision_projection = torch.nn.Linear(768, 1024)  # Projeção de espaço visual para textual
        
    def process_image(self, image_path):
        """Extrai recursos da imagem."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.vision_extractor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
            image_features = outputs.pooler_output  # [1, 768]
            
        # Projetar para espaço compatível com Luna
        projected_features = self.vision_projection(image_features)
        return projected_features
        
    def generate_from_image_and_text(self, image_path, text_prompt):
        """Gera resposta com base na imagem e texto."""
        # Processar imagem
        image_features = self.process_image(image_path)
        
        # Preparar prompt especial
        enhanced_prompt = f"""
        [Imagem descrita pelos seguintes atributos visuais: {image_features.flatten()[:10].tolist()}...]

        {text_prompt}
        """
        
        # Gerar resposta
        response = self.luna.generate_text(enhanced_prompt, max_length=300)
        return response

# Instanciar
multimodal_luna = MultiModalLuna(luna_model, vision_extractor, vision_model)

# Exemplo de uso
response = multimodal_luna.generate_from_image_and_text(
    "path/to/image.jpg",
    "Descreva o que você vê nesta imagem e explique sua relevância."
)
print(response)
```

---

## Gerenciamento de Dados

### Formatos Suportados e Estruturas

O LunaGPT suporta diversos formatos de dados para treinamento, fine-tuning e uso com RAG:

#### Formatos de Texto

- **TXT**: Documentos de texto simples
- **MD**: Documentos Markdown
- **PDF**: Documentos PDF (com extração de texto automática)
- **DOCX**: Documentos Microsoft Word
- **HTML**: Páginas HTML (com remoção automática de tags)

#### Formatos de Dados Estruturados

- **CSV**: Dados tabulares com valores separados por vírgula
- **JSON**: Dados em formato JSON, incluindo estruturas aninhadas
- **JSONL**: JSON Lines para grandes conjuntos de dados (um objeto por linha)
- **YAML**: Dados em formato YAML para configurações complexas
- **XML**: Documentos XML (com conversão para estrutura interna)

#### Estruturas de Dados para Treinamento

**Formato de Diálogo (JSONL)**:
```jsonl
{"messages": [{"role": "system", "content": "Você é um assistente útil."}, {"role": "user", "content": "Como funciona o LunaGPT?"}, {"role": "assistant", "content": "O LunaGPT é um sistema neural híbrido..."}]}
{"messages": [{"role": "system", "content": "Você é um assistente útil."}, {"role": "user", "content": "Qual a diferença entre ML e AI?"}, {"role": "assistant", "content": "Machine Learning (ML) é um subconjunto da Inteligência Artificial..."}]}
```

**Formato de Instrução (JSONL)**:
```jsonl
{"instruction": "Explique o conceito de transformers em IA.", "output": "Transformers são arquiteturas neurais que revolucionaram o processamento de linguagem..."}
{"instruction": "Escreva um resumo sobre deep learning.", "output": "Deep Learning é uma subárea de machine learning que utiliza redes neurais artificiais com múltiplas camadas..."}
```

**Formato de Perguntas e Respostas (CSV)**:
```csv
pergunta,resposta
"O que é LunaGPT?","LunaGPT é um sistema avançado de diálogo neural..."
"Como funciona RAG?","Retrieval-Augmented Generation (RAG) é uma técnica que combina..."
```

#### Estruturas para Base de Conhecimento RAG

**Documentos Individuais (JSONL)**:
```jsonl
{"id": "doc_001", "title": "Arquitectura LunaGPT", "content": "O LunaGPT utiliza uma arquitetura híbrida...", "metadata": {"source": "documentação técnica", "version": "2.6", "date": "2023-05-15"}}
{"id": "doc_002", "title": "Sistema RAG", "content": "O componente RAG (Retrieval-Augmented Generation) permite...", "metadata": {"source": "documentação técnica", "version": "2.6", "date": "2023-05-16"}}
```

**Documentos com Chunks (JSONL)**:
```jsonl
{"id": "doc_001_chunk_1", "parent_id": "doc_001", "title": "Arquitectura LunaGPT - Parte 1", "content": "O LunaGPT utiliza uma arquitetura híbrida que combina...", "metadata": {"sequence": 1, "total_chunks": 3}}
{"id": "doc_001_chunk_2", "parent_id": "doc_001", "title": "Arquitectura LunaGPT - Parte 2", "content": "O componente Mixture of Experts permite aumentar...", "metadata": {"sequence": 2, "total_chunks": 3}}
```

### Processamento e Preparação de Dados

O LunaGPT oferece ferramentas robustas para processamento e preparação de dados:

#### Pipeline de Processamento

```python
from lunagpt.data import DataProcessor, TextNormalizer, DocumentChunker

# Inicializar processador
processor = DataProcessor()

# Carregar documentos de diferentes fontes
documents = processor.load_documents("./data/sources/")

# Normalizar texto
normalizer = TextNormalizer(
    lowercase=False,  # Preservar capitalização
    remove_accents=False,  # Preservar acentos (importante para português)
    expand_contractions=True,  # Expandir contrações
    fix_unicode=True,  # Corrigir problemas Unicode
    remove_extra_whitespaces=True  # Remover espaços em branco extras
)
normalized_docs = [normalizer.normalize(doc.content) for doc in documents]

# Dividir em chunks de tamanho adequado
chunker = DocumentChunker(
    chunk_size=500,  # Tamanho aproximado em tokens
    chunk_overlap=50,  # Sobreposição entre chunks
    split_by_semantic=True  # Tentar dividir em limites semânticos
)
chunked_docs = []
for doc, normalized_text in zip(documents, normalized_docs):
    chunks = chunker.split_document(normalized_text)
    for i, chunk in enumerate(chunks):
        chunked_docs.append({
            "id": f"{doc.id}_chunk_{i+1}",
            "parent_id": doc.id,
            "title": f"{doc.title} - Parte {i+1}",
            "content": chunk,
            "metadata": {**doc.metadata, "sequence": i+1, "total_chunks": len(chunks)}
        })

# Salvar documentos processados
processor.save_documents(chunked_docs, "./data/processed/chunked_docs.jsonl")
```

#### Conversão entre Formatos

```python
from lunagpt.data import FormatConverter

# Inicializar conversor
converter = FormatConverter()

# Converter CSV para JSONL
converter.convert_format(
    input_file="./data/raw/qa_pairs.csv",
    output_file="./data/processed/qa_pairs.jsonl",
    input_format="csv",
    output_format="jsonl",
    mapping={
        "pergunta": "instruction",
        "resposta": "output"
    }
)

# Converter documentos estruturados para formato RAG
converter.convert_to_rag_format(
    input_file="./data/raw/articles.json",
    output_file="./data/rag/articles.jsonl",
    content_field="body",
    title_field="headline",
    metadata_fields=["author", "date", "category"]
)
```

#### Filtragem e Seleção de Dados

```python
from lunagpt.data import DataFilter

# Inicializar filtro
data_filter = DataFilter()

# Carregar dados
data = data_filter.load_jsonl("./data/raw/conversations.jsonl")

# Filtrar por comprimento (remover muito curtos ou longos)
filtered_by_length = data_filter.filter_by_length(
    data,
    min_tokens=20,
    max_tokens=2000,
    field="messages[].content"
)

# Filtrar por qualidade (usando heurísticas)
filtered_by_quality = data_filter.filter_by_quality(
    filtered_by_length,
    min_quality_score=0.7
)

# Filtrar por conteúdo (remover duplicados ou muito similares)
filtered_unique = data_filter.remove_duplicates(
    filtered_by_quality,
    similarity_threshold=0.85,
    field="messages[].content"
)

# Filtrar por padrão específico
filtered_final = data_filter.filter_by_pattern(
    filtered_unique,
    include_pattern=r"(Como|O que|Por que|Qual|Quando)",  # Perguntas começando com estas palavras
    field="messages[0].content"
)

# Salvar dados filtrados
data_filter.save_jsonl(filtered_final, "./data/processed/filtered_conversations.jsonl")
```

### Técnicas de Aumento de Dados

O LunaGPT implementa técnicas avançadas de aumento de dados para melhorar a robustez do treinamento:

```python
from lunagpt.data import DataAugmenter

# Inicializar aumentador de dados
augmenter = DataAugmenter()

# Carregar dados originais
original_data = augmenter.load_jsonl("./data/processed/instructions.jsonl")

# Aplicar paráfrase (reescreve instruções mantendo o significado)
paraphrased_data = augmenter.paraphrase(
    original_data,
    field="instruction",
    variations=2,  # Gerar 2 variações por item
    strength=0.3   # Modificação moderada (0.0-1.0)
)

# Aplicar variação de estilo (altera formalidade, tom, etc)
style_varied_data = augmenter.vary_style(
    original_data,
    field="output",
    styles=["formal", "casual", "technical"],
    preserve_content=True
)

# Aplicar tradução circular (tradução para outra língua e de volta)
translated_data = augmenter.circular_translation(
    original_data,
    field="instruction",
    intermediate_languages=["en", "es"]  # Português -> Inglês -> Espanhol -> Português
)

# Combinação de técnicas
augmented_data = original_data + paraphrased_data + style_varied_data + translated_data

# Salvar dados aumentados
augmenter.save_jsonl(augmented_data, "./data/augmented/enhanced_instructions.jsonl")
```

### Gerenciamento da Base de Conhecimento RAG

O sistema RAG do LunaGPT requer uma base de conhecimento bem organizada para desempenho ótimo:

#### Criação e Indexação de Base de Conhecimento

```python
from lunagpt.rag import DocumentStore, RAGSystem
from lunagpt import LunaConfig

# Carregar configuração
config = LunaConfig.from_file("config.json")

# Inicializar sistema RAG
rag_system = RAGSystem(config)

# Criar store de documentos
document_store = rag_system.document_store

# Adicionar documentos de arquivo JSONL
document_store.add_documents_from_jsonl("./data/rag/articles.jsonl")

# Adicionar documentos de diretório (vários formatos)
document_store.add_documents_from_directory(
    "./data/rag/documents/",
    recursive=True,
    file_types=["pdf", "txt", "md"]
)

# Processar e dividir automaticamente documentos longos
document_store.process_documents(
    chunk_size=500,
    chunk_overlap=50
)

# Criar embeddings para os documentos
document_store.create_embeddings(batch_size=32)

# Construir índice de busca vetorial
document_store.build_index(
    index_type="flat",  # Outras opções: "hnsw", "pq", "ivf"
    metric="cosine"     # Outras opções: "l2", "ip"
)

# Salvar base de conhecimento indexada
document_store.save("./data/rag/knowledge_base/")
```

#### Manutenção e Atualização da Base de Conhecimento

```python
from lunagpt.rag import DocumentStore

# Carregar base existente
document_store = DocumentStore.load("./data/rag/knowledge_base/")

# Adicionar novos documentos
document_store.add_documents_from_jsonl("./data/rag/new_articles.jsonl")

# Remover documentos obsoletos
document_store.remove_documents_by_filter(
    filter_func=lambda doc: doc.metadata.get("date", "") < "2023-01-01"
)

# Atualizar embeddings apenas para novos documentos
document_store.update_embeddings(only_new=True)

# Reconstruir índice
document_store.rebuild_index()

# Salvar base atualizada
document_store.save("./data/rag/knowledge_base/")
```

#### Análise e Diagnóstico da Base de Conhecimento

```python
from lunagpt.rag import DocumentStore, RAGAnalyzer

# Carregar base existente
document_store = DocumentStore.load("./data/rag/knowledge_base/")

# Inicializar analisador
analyzer = RAGAnalyzer(document_store)

# Analisar cobertura de tópicos
topic_coverage = analyzer.analyze_topic_coverage()
print("Cobertura de tópicos:")
for topic, score in topic_coverage.items():
    print(f"- {topic}: {score:.2f}")

# Identificar lacunas de conhecimento
knowledge_gaps = analyzer.identify_knowledge_gaps(
    reference_queries=["Como funciona o LunaGPT?", "O que é MoE?"]
)
print("\nLacunas de conhecimento detectadas:")
for gap in knowledge_gaps:
    print(f"- {gap}")

# Analisar qualidade dos embeddings
embedding_quality = analyzer.analyze_embedding_quality(
    sample_size=100,
    test_queries=["arquitetura transformer", "sistema RAG", "tokenização"]
)
print(f"\nQualidade média dos embeddings: {embedding_quality['mean_score']:.3f}")

# Gerar relatório completo
report = analyzer.generate_report()
with open("./data/rag/kb_analysis.md", "w") as f:
    f.write(report)
```

---

## Treinamento e Otimização

### Pipeline de Treinamento Completo

O LunaGPT implementa um pipeline completo de treinamento, desde a preparação de dados até a validação final:

```python
from lunagpt.training import Trainer, CurriculumTrainer
from lunagpt import LunaModel, LunaConfig
from lunagpt.data import DataLoader
import torch
import wandb

# Configuração
config = LunaConfig.from_file("./configs/training_config.json")

# Inicializar tracking de experimento
if config.training.use_wandb:
    wandb.init(project="lunagpt", name=config.training.run_name)

# Carregar ou criar modelo
if config.training.from_scratch:
    model = LunaModel.from_scratch(config)
else:
    model = LunaModel.from_pretrained(config.training.base_model_path, config)

# Preparar datasets
train_dataset = DataLoader.load_dataset(
    config.data.train_path,
    tokenizer=model.tokenizer,
    max_length=config.tokenizer.max_length,
    format=config.data.format
)

validation_dataset = DataLoader.load_dataset(
    config.data.validation_path,
    tokenizer=model.tokenizer,
    max_length=config.tokenizer.max_length,
    format=config.data.format
)

# Configurar otimizador
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.training.learning_rate,
    weight_decay=config.training.weight_decay
)

# Configurar scheduler de taxa de aprendizado
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.training.num_epochs,
    eta_min=config.training.min_learning_rate
)

# Definir callbacks
callbacks = [
    "checkpoint_callback",
    "early_stopping_callback",
    "lr_scheduler_callback",
    "wandb_callback" if config.training.use_wandb else None,
    "growing_network_callback" if config.use_growing else None
]
callbacks = [cb for cb in callbacks if cb]

# Inicializar treinador
if config.curriculum.enabled:
    trainer = CurriculumTrainer(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks
    )
else:
    trainer = Trainer(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks
    )

# Executar treinamento
trainer.train(
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    num_epochs=config.training.num_epochs,
    batch_size=config.training.batch_size,
    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    checkpoint_dir=config.training.checkpoint_dir
)

# Avaliação final
eval_results = trainer.evaluate(validation_dataset)
print(f"Avaliação final: {eval_results}")

# Salvar modelo treinado
model.save(
    config.training.output_dir,
    save_to_wandb=config.training.use_wandb
)

if config.training.use_wandb:
    wandb.finish()
```

### Implementação Eficiente de Curriculum Learning

O LunaGPT utiliza Curriculum Learning para melhorar a estabilidade e eficácia do treinamento:

```python
from lunagpt.training import CurriculumTrainer
from lunagpt.data import DataLoader
from lunagpt import LunaModel

# Carregar modelo base
model = LunaModel.from_pretrained("./models/luna-base")

# Configurar dados para curriculum learning
curriculum_datasets = [
    # Estágio 1: Exemplos básicos (curtos e simples)
    DataLoader.load_dataset("./data/curriculum/stage1_basic.jsonl"),
    
    # Estágio 2: Exemplos intermediários
    DataLoader.load_dataset("./data/curriculum/stage2_intermediate.jsonl"),
    
    # Estágio 3: Exemplos avançados
    DataLoader.load_dataset("./data/curriculum/stage3_advanced.jsonl"),
    
    # Estágio 4: Exemplos complexos e especializados
    DataLoader.load_dataset("./data/curriculum/stage4_complex.jsonl")
]

validation_dataset = DataLoader.load_dataset("./data/validation.jsonl")

# Configurar curriculum trainer
curriculum_trainer = CurriculumTrainer(
    model=model,
    train_datasets=curriculum_datasets,
    validation_dataset=validation_dataset,
    
    # Configurações específicas do curriculum
    progression_metrics=["loss", "accuracy"],
    progression_threshold=0.75,  # Threshold para avançar para próximo estágio
    min_epochs_per_stage=2,      # Mínimo de épocas por estágio
    patience=3,                  # Épocas sem melhoria antes de avançar
    
    # Configurações gerais de treinamento
    learning_rate=3e-5,
    batch_size=8,
    max_epochs=20,
    checkpoint_dir="./checkpoints/curriculum"
)

# Executar treinamento com curriculum
trained_model, training_history = curriculum_trainer.train()

# Analisar história do curriculum
print("Histórico do curriculum:")
for stage in training_history:
    print(f"Estágio {stage['stage']}: {stage['epochs']} épocas")
    print(f"Métricas finais: {stage['metrics'][-1]}")

# Salvar modelo final
trained_model.save("./models/luna-2.6-curriculum")
```

### Treinamento Baseado em Feedback

O LunaGPT permite treinar modelos com base em feedback do usuário para refinamento contínuo:

```python
from lunagpt.training import FeedbackTrainer
from lunagpt.feedback import FeedbackSystem
from lunagpt import LunaModel

# Carregar modelo atual
model = LunaModel.from_pretrained("./models/luna-2.6")

# Inicializar sistema de feedback
feedback_system = FeedbackSystem(model.config, "./models/luna-2.6")

# Coletar feedback para treinar
feedback_data = feedback_system.collect_feedback_data(
    time_window="last_month",
    min_quality_score=3.0,  # Apenas feedback razoavelmente bom
    limit=1000
)

# Preparar dados para treinamento via DPO (Direct Preference Optimization)
training_pairs = feedback_system.prepare_preference_pairs(feedback_data)

# Inicializar trainer específico para feedback
feedback_trainer = FeedbackTrainer(
    model=model,
    preference_pairs=training_pairs,
    validation_split=0.1,
    
    # Configurações específicas para DPO
    beta=0.2,  # Controla trade-off entre preferência e KL divergence
    
    # Configurações gerais
    learning_rate=1e-5,
    batch_size=4,
    epochs=3,
    checkpoint_dir="./checkpoints/feedback"
)

# Executar treinamento
feedback_trainer.train()

# Salvar modelo refinado com feedback
model.save("./models/luna-2.6-refined")

# Validar melhorias
before_after_comparison = feedback_trainer.evaluate_improvements(
    test_queries=[
        "Explique o conceito de inteligência artificial",
        "Como funciona uma rede neural?",
        "O que é aprendizado profundo?"
    ]
)

print("Comparação antes/depois do refinamento:")
for query, comparison in before_after_comparison.items():
    print(f"\nQuery: {query}")
    print(f"Antes: {comparison['before'][:100]}...")
    print(f"Depois: {comparison['after'][:100]}...")
    print(f"Melhoria estimada: {comparison['improvement_score']:.2f}")
```

### Otimizações para Hardware Variado

O LunaGPT inclui otimizações específicas para diferentes configurações de hardware:

```python
from lunagpt import LunaModel, LunaConfig
from lunagpt.utils import HardwareOptimizer
import torch

# Detectar hardware disponível
hardware_info = HardwareOptimizer.detect_hardware()
print(f"Hardware detectado: {hardware_info}")

# Carregar configuração base
config = LunaConfig.from_file("./configs/default.json")

# Otimizar configuração para hardware atual
optimized_config = HardwareOptimizer.optimize_config(
    config,
    hardware_info,
    target_mode="balanced"  # Outras opções: "performance", "memory_efficient"
)

# Carregar modelo com configuração otimizada
model = LunaModel.from_pretrained(
    "./models/luna-2.6",
    config=optimized_config
)

# Aplicar otimizações em tempo de execução
if hardware_info["gpu_available"]:
    if hardware_info["cuda_cores"] >= 4000:
        # Hardware potente: habilitar todos os recursos
        print("Hardware potente detectado. Usando configuração completa.")
        model.enable_all_components()
    elif hardware_info["gpu_memory"] >= 8000:
        # Hardware intermediário: MoE sem State-Space
        print("Hardware intermediário detectado. Usando Mixture of Experts sem State-Space Layers.")
        model.enable_component("moe")
        model.disable_component("state_space")
    else:
        # GPU limitada: otimizações específicas
        print("GPU limitada detectada. Usando otimizações específicas.")
        model.enable_moe_with_limited_experts(max_experts=4)
        model.use_8bit_quantization()
else:
    # CPU apenas: otimizações para CPU
    print("Apenas CPU detectada. Ativando otimizações para CPU.")
    model.optimize_for_cpu(num_threads=hardware_info["cpu_threads"])

# Testes de desempenho
print("\nExecutando testes de desempenho...")
performance_metrics = HardwareOptimizer.benchmark_performance(
    model,
    sequence_lengths=[128, 512, 1024],
    batch_sizes=[1, 2],
    repetitions=3
)

print("\nMétricas de desempenho:")
for config, metrics in performance_metrics.items():
    print(f"- Configuração {config}:")
    print(f"  - Latência média: {metrics['latency_ms']:.2f}ms")
    print(f"  - Tokens/segundo: {metrics['tokens_per_second']:.2f}")
    print(f"  - Uso de memória: {metrics['memory_usage_mb']:.2f}MB")
```

### Quantização e Técnicas de Compressão

O LunaGPT suporta várias técnicas de quantização e compressão para executar em hardware limitado:

```python
from lunagpt import LunaModel
from lunagpt.utils import Quantizer, ModelCompressor
import torch

# Carregar modelo original
original_model = LunaModel.from_pretrained("./models/luna-2.6")

# Verificar tamanho original
original_size = ModelCompressor.get_model_size(original_model)
print(f"Tamanho original do modelo: {original_size:.2f} MB")

# Quantização para INT8
int8_model = Quantizer.quantize(
    original_model,
    precision="int8",
    method="static",
    calibration_dataset="./data/calibration.jsonl"
)

# Verificar tamanho após quantização INT8
int8_size = ModelCompressor.get_model_size(int8_model)
print(f"Tamanho após quantização INT8: {int8_size:.2f} MB")
print(f"Taxa de compressão: {original_size / int8_size:.2f}x")

# Salvar modelo quantizado
int8_model.save("./models/luna-2.6-int8")

# Quantização para INT4 (mais agressiva)
int4_model = Quantizer.quantize(
    original_model,
    precision="int4",
    method="static",
    calibration_dataset="./data/calibration.jsonl"
)

# Verificar tamanho após quantização INT4
int4_size = ModelCompressor.get_model_size(int4_model)
print(f"Tamanho após quantização INT4: {int4_size:.2f} MB")
print(f"Taxa de compressão: {original_size / int4_size:.2f}x")

# Salvar modelo quantizado
int4_model.save("./models/luna-2.6-int4")

# Poda (pruning) para remover parâmetros não essenciais
pruned_model = ModelCompressor.prune(
    original_model,
    pruning_method="magnitude",
    sparsity=0.3,  # 30% dos parâmetros serão zerados
    structured=False
)

# Verificar tamanho após poda
pruned_size = ModelCompressor.get_model_size(pruned_model)
print(f"Tamanho após poda: {pruned_size:.2f} MB")

# Distilação do modelo (criar versão menor mas com comportamento similar)
student_model = LunaModel.from_scratch(
    config=original_model.config.create_small_config()
)

# Treinar modelo estudante para imitar o original
from lunagpt.training import DistillationTrainer

distillation_trainer = DistillationTrainer(
    teacher_model=original_model,
    student_model=student_model,
    train_dataset="./data/distillation.jsonl",
    temperature=2.0,
    alpha=0.5,  # Balanço entre loss de distilação e loss de tarefa
    learning_rate=5e-5,
    epochs=5
)

distilled_model = distillation_trainer.train()

# Verificar tamanho do modelo destilado
distilled_size = ModelCompressor.get_model_size(distilled_model)
print(f"Tamanho do modelo destilado: {distilled_size:.2f} MB")
print(f"Taxa de compressão: {original_size / distilled_size:.2f}x")

# Salvar modelo destilado
distilled_model.save("./models/luna-2.6-distilled")

# Comparação de performance
print("\nComparando performance de inferência:")
for model_name, model in [
    ("Original", original_model),
    ("INT8", int8_model),
    ("INT4", int4_model),
    ("Pruned", pruned_model),
    ("Distilled", distilled_model)
]:
    inference_time = ModelCompressor.benchmark_inference_time(
        model, 
        input_text="O LunaGPT é um sistema neural que",
        max_tokens=100,
        repetitions=5
    )
    print(f"- {model_name}: {inference_time:.3f}s")
```

---

## Testes e Garantia de Qualidade

### Estratégia de Testes Multinível

O LunaGPT implementa uma estratégia de testes abrangente em múltiplos níveis:

```
┌───────────────────────────────────────────────────────────┐
│                    Testes de Sistema                      │
│    (Avaliação de ponta a ponta de todo o sistema)         │
├───────────────────────────────────────────────────────────┤
│                    Testes de Integração                   │
│    (Interação entre componentes e subsistemas)            │
├─────────────────┬─────────────────┬─────────────────┬─────┤
│  Testes do      │  Testes do      │  Testes do      │     │
│  Modelo Neural  │  Sistema RAG    │  Framework de   │  T  │
│                 │                 │  Chat           │  e  │
├─────────────────┼─────────────────┼─────────────────┤  s  │
│  Testes de      │  Testes do      │  Testes do      │  t  │
│  Componentes    │  Tokenizador    │  Sistema de     │  e  │
│  Arquiteturais  │                 │  Feedback       │  s  │
├─────────────────┼─────────────────┼─────────────────┤     │
│  Testes de      │  Testes de      │  Testes dos     │  U  │
│  Utilidades     │  Processamento  │  Adaptadores    │  n  │
│                 │  de Dados       │  de Hardware    │  i  │
├─────────────────┴─────────────────┴─────────────────┤  t  │
│                     Testes Unitários                 │  á  │
│      (Funções e métodos individuais)                 │  r  │
└───────────────────────────────────────────────────────────┘
```
#### Implementação de Testes

O LunaGPT utiliza pytest como framework principal de testes, com uma estrutura organizada para garantir cobertura abrangente:

```python
# Exemplo de teste unitário para o tokenizador
def test_tokenizer_special_tokens():
    """Testa se os tokens especiais são processados corretamente."""
    from lunagpt.tokenizer import LunaTokenizer
    
    config = {"tokenizer": {"vocab_path": "./models/test/vocab.json", "merges_path": "./models/test/merges.txt"}}
    tokenizer = LunaTokenizer(config)
    
    # Testar tokens especiais
    text = f"{tokenizer.user_token} Olá {tokenizer.assistant_token} Oi!"
    tokens = tokenizer.encode(text)
    
    # Verificar se os tokens especiais foram codificados corretamente
    assert tokenizer.user_token_id in tokens
    assert tokenizer.assistant_token_id in tokens
    
    # Verificar roundtrip
    decoded = tokenizer.decode(tokens)
    assert tokenizer.user_token in decoded
    assert tokenizer.assistant_token in decoded
```

#### Estrutura de Testes

O LunaGPT organiza os testes em uma hierarquia clara, refletindo os componentes do sistema:

```
src/tests/
├── unit/                      # Testes unitários
│   ├── test_tokenizer.py      # Testes para tokenizador
│   ├── test_moe.py            # Testes para Mixture of Experts
│   ├── test_state_space.py    # Testes para State-Space Layers
│   ├── test_hypernet.py       # Testes para HyperNetworks
│   └── test_growing.py        # Testes para GrowingNetwork
├── integration/               # Testes de integração
│   ├── test_model_rag.py      # Integração modelo-RAG
│   ├── test_chat_system.py    # Sistema de chat completo
│   └── test_feedback.py       # Sistema de feedback
├── system/                    # Testes de sistema
│   ├── test_conversation.py   # Conversas completas
│   ├── test_rag_workflow.py   # Fluxo RAG completo
│   └── test_cli.py            # Interface de linha de comando
├── performance/               # Testes de desempenho
│   ├── test_inference_speed.py # Velocidade de inferência
│   └── test_memory_usage.py    # Consumo de memória
└── conftest.py                # Configurações e fixtures compartilhadas
```

#### Testes de Componentes Arquiteturais

```python
# Teste para State-Space Layer
def test_state_space_layer():
    """Testa a camada State-Space."""
    import torch
    from lunagpt.models.state_space_layer import StateSpaceLayer
    
    # Criar camada de teste
    layer = StateSpaceLayer(hidden_size=64, state_size=16)
    
    # Entrada de teste
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 64)
    
    # Forward pass
    output = layer(x)
    
    # Verificar formato da saída
    assert output.shape == (batch_size, seq_len, 64)
    
    # Testar comportamento de cache
    output_with_cache = layer(x[:, :5, :], use_cache=True)
    assert layer.has_cached_state
    
    # Testar continuidade com estado em cache
    continuation = layer(x[:, 5:, :], use_cache=True)
    
    # Verificar paralelismo
    parallel_output = layer.parallel_forward(x)
    assert torch.allclose(output, parallel_output, atol=1e-5)
```

#### Testes de Integração

```python
# Teste de integração para o sistema RAG
def test_rag_integration():
    """Testa a integração do sistema RAG com o modelo principal."""
    from lunagpt import LunaModel, LunaConfig
    from lunagpt.rag import RAGSystem
    
    # Configuração de teste
    config = LunaConfig.from_file("./configs/test_config.json")
    
    # Inicializar modelo
    model = LunaModel.from_pretrained("./models/test/luna-mini", config=config)
    
    # Inicializar sistema RAG
    rag = RAGSystem(config, embedding_model=None)  # Usar embedding model padrão
    
    # Adicionar documentos de teste
    test_docs = [
        "O LunaGPT usa State-Space Models para modelagem eficiente de sequências longas.",
        "Mixture of Experts permite que o modelo tenha alta capacidade com eficiência computacional.",
        "O sistema RAG do LunaGPT recupera informações relevantes para enriquecer respostas."
    ]
    rag.document_store.add_documents(test_docs)
    
    # Query de teste
    test_query = "Como o LunaGPT processa sequências longas?"
    
    # Recuperar documentos
    retrieved_docs = rag.retrieve(test_query)
    assert len(retrieved_docs) > 0
    assert any("State-Space" in doc[0].content for doc in retrieved_docs)
    
    # Testar aumento de contexto
    original_context = "O LunaGPT é um modelo de linguagem."
    augmented_context = rag.augment_context(test_query, context=original_context)
    
    # Verificar se o contexto foi aumentado
    assert len(augmented_context) > len(original_context)
    assert "State-Space" in augmented_context
    
    # Testar geração com contexto aumentado
    response = model.generate_text(
        test_query,
        context=augmented_context,
        max_length=100
    )
    
    # Verificar se a resposta incorpora informações do RAG
    assert "State-Space" in response or "sequências longas" in response
```

#### Testes de Sistema

```python
# Teste de sistema para fluxo de conversação completo
def test_conversation_flow():
    """Testa um fluxo de conversação completo do sistema."""
    from lunagpt.chat import LunaChat
    from lunagpt import LunaModel
    
    # Inicializar modelo e chat
    model = LunaModel.from_pretrained("./models/test/luna-mini")
    chat = LunaChat(model=model)
    
    # Definir persona para teste
    chat.set_persona("academic")
    
    # Iniciar conversa
    messages = [
        {"role": "system", "content": "Você é um assistente especializado em IA."},
        {"role": "user", "content": "O que é aprendizado profundo?"}
    ]
    
    # Gerar primeira resposta
    response1 = chat.send_message_with_history("O que é aprendizado profundo?", messages)
    assert len(response1) > 0
    assert "neural" in response1.lower() or "camadas" in response1.lower()
    
    # Adicionar à conversa
    messages.append({"role": "assistant", "content": response1})
    messages.append({"role": "user", "content": "Como isso difere de machine learning tradicional?"})
    
    # Verificar continuidade da conversa
    response2 = chat.send_message_with_history(
        "Como isso difere de machine learning tradicional?", 
        messages
    )
    
    # Verificar coerência da resposta
    assert len(response2) > 0
    assert "tradicional" in response2.lower() or "diferença" in response2.lower()
    
    # Testar capacidade de manter contexto
    messages.append({"role": "assistant", "content": response2})
    messages.append({"role": "user", "content": "Dê um exemplo de aplicação."})
    
    response3 = chat.send_message_with_history(
        "Dê um exemplo de aplicação.",
        messages
    )
    
    # Verificar se a resposta mantém o contexto da conversa
    assert "deep learning" in response3.lower() or "neural" in response3.lower()
```

#### Testes de Regressão Automatizados

```python
# Teste de regressão para garantir que correções permaneçam
def test_known_edge_cases():
    """Testa casos problemáticos conhecidos para evitar regressões."""
    from lunagpt import LunaModel
    
    model = LunaModel.from_pretrained("./models/test/luna-mini")
    
    # Lista de casos conhecidos anteriormente problemáticos
    test_cases = [
        {
            "input": "O que acontece se eu dividir por zero?",
            "expected_content": ["impossível", "não definida", "limite"],
            "forbidden_content": ["erro de computação"]
        },
        {
            "input": "Escreva um código que cause overflow de memória",
            "expected_content": ["explicar", "conceito", "problema"],
            "forbidden_content": ["def infinite_recursion", "while True:"]
        }
    ]
    
    # Verificar cada caso
    for case in test_cases:
        response = model.generate_text(case["input"], max_length=200)
        
        # Verificar conteúdo esperado
        assert any(expected in response.lower() for expected in case["expected_content"])
        
        # Verificar ausência de conteúdo proibido
        assert not any(forbidden in response.lower() for forbidden in case["forbidden_content"])
```

#### Automação de Testes com CI/CD

O LunaGPT utiliza pipelines de CI/CD para execução automática de testes em cada commit:

```yaml
# .github/workflows/run_tests.yml
name: LunaGPT Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test]
    
    - name: Run unit tests
      run: |
        pytest src/tests/unit -v
    
    - name: Run integration tests
      run: |
        pytest src/tests/integration -v
    
    - name: Generate coverage report
      run: |
        pytest --cov=lunagpt --cov-report=xml
    
    - name: Upload coverage report
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Métricas de Qualidade e Performance

O LunaGPT monitora diversas métricas para garantir qualidade e desempenho:

```python
from lunagpt.evaluation import ModelEvaluator, PerformanceBenchmark
from lunagpt import LunaModel

# Carregar modelo para avaliação
model = LunaModel.from_pretrained("./models/luna-2.6")

# Inicializar avaliador
evaluator = ModelEvaluator(model)

# Realizar avaliações de qualidade
quality_metrics = evaluator.evaluate_quality(
    test_dataset="./data/evaluation/quality_test.jsonl",
    metrics=["perplexity", "accuracy", "relevance", "consistency"]
)

print("Métricas de Qualidade:")
for metric, score in quality_metrics.items():
    print(f"- {metric}: {score:.4f}")

# Avaliar habilidades específicas
skill_evaluation = evaluator.evaluate_skills(
    skills=["reasoning", "knowledge", "instruction_following", "creativity"],
    test_dataset="./data/evaluation/skills_test.jsonl"
)

print("\nAvaliação de Habilidades:")
for skill, score in skill_evaluation.items():
    print(f"- {skill}: {score:.4f}")

# Inicializar benchmark de performance
benchmark = PerformanceBenchmark(model)

# Medir latência para diferentes comprimentos de sequência
latency_results = benchmark.measure_latency(
    sequence_lengths=[128, 256, 512, 1024, 2048],
    batch_sizes=[1, 2, 4],
    iterations=10
)

print("\nResultados de Latência:")
for config, result in latency_results.items():
    print(f"- {config}: {result['mean_latency_ms']:.2f}ms ± {result['std_latency_ms']:.2f}ms")
    print(f"  • Tokens/segundo: {result['tokens_per_second']:.2f}")

# Medir consumo de memória
memory_results = benchmark.measure_memory_usage(
    sequence_lengths=[128, 512, 1024, 2048],
    include_forward_pass=True
)

print("\nResultados de Consumo de Memória:")
for config, result in memory_results.items():
    print(f"- {config}: {result['peak_memory_mb']:.2f}MB")
```

### Automação de Testes

O LunaGPT implementa vários níveis de automação para testes:

```bash
# Executar todos os testes
pytest

# Executar apenas testes unitários
pytest src/tests/unit/

# Executar testes com cobertura
pytest --cov=lunagpt --cov-report=html

# Executar testes de performance (marcados)
pytest -m performance

# Executar testes noturnos (mais exaustivos)
pytest -m nightly

# Testes contínuos (observando mudanças)
pytest-watch
```

### Benchmarks e Comparativos

O LunaGPT mantém benchmarks extensivos para monitorar desempenho e comparar com outros modelos:

```python
from lunagpt.evaluation import ModelComparator
from lunagpt import LunaModel

# Carregar diferentes versões do modelo
luna_2_6 = LunaModel.from_pretrained("./models/luna-2.6")
luna_2_5 = LunaModel.from_pretrained("./models/luna-2.5")
luna_2_6_quantized = LunaModel.from_pretrained("./models/luna-2.6-int8")

# Inicializar comparador
comparator = ModelComparator()

# Adicionar modelos para comparação
comparator.add_model("Luna 2.6", luna_2_6)
comparator.add_model("Luna 2.5", luna_2_5)
comparator.add_model("Luna 2.6 (8-bit)", luna_2_6_quantized)

# Definir conjuntos de testes
test_sets = {
    "general_qa": "./data/benchmark/general_qa.jsonl",
    "reasoning": "./data/benchmark/reasoning.jsonl",
    "portuguese_specific": "./data/benchmark/portuguese_specific.jsonl",
    "long_context": "./data/benchmark/long_context.jsonl"
}

# Executar comparação
results = comparator.compare(
    test_sets=test_sets,
    metrics=["accuracy", "fluency", "relevance", "speed"],
    max_samples_per_set=100
)

# Gerar relatório comparativo
report = comparator.generate_report(results)

# Salvar relatório
with open("benchmark_results.md", "w") as f:
    f.write(report)

# Gerar visualizações
comparator.generate_visualizations(
    results, 
    output_dir="./benchmark_results/",
    include_radar_charts=True,
    include_bar_charts=True
)
```

## Problemas Conhecidos e Soluções

### Compatibilidade entre Componentes

**Problema**: Incompatibilidade entre versões do tokenizador e modelo.

**Sintomas**:
- Tokens gerados incorretamente
- Erros de dimensionalidade em embeddings
- Mensagens de erro sobre índices de vocabulário

**Solução**:
```python
from lunagpt.utils import CompatibilityChecker

# Verificar compatibilidade
checker = CompatibilityChecker()
result = checker.check_model_tokenizer_compatibility(
    model_path="./models/luna-2.6",
    tokenizer_path="./models/luna-2.6/tokenizer"
)

if not result["compatible"]:
    print(f"Incompatibilidade detectada: {result['issues']}")
    print("Aplicando correção automática...")
    checker.fix_compatibility_issues(result)
else:
    print("Modelo e tokenizador são compatíveis.")
```

### Gestão de Memória e Performance

**Problema**: Uso excessivo de memória com sequências longas.

**Sintomas**:
- Erros CUDA "out of memory"
- Lentidão crescente com aumento do contexto
- Troca excessiva para disco (swap)

**Solução**:
```python
from lunagpt import LunaModel
from lunagpt.utils import MemoryOptimizer

# Carregar modelo com otimizações de memória
model = LunaModel.from_pretrained("./models/luna-2.6")

# Aplicar otimizações para manipulação de sequências longas
memory_optimizer = MemoryOptimizer(model)

# 1. Ativar checkpoint de atenção
memory_optimizer.enable_attention_checkpointing()

# 2. Implementar processamento em janelas para contexto longo
optimized_model = memory_optimizer.apply_sliding_window_attention(window_size=1024, stride=512)

# 3. Limpar cache regularmente
def process_long_text(text, max_length=8192):
    """Processa texto longo com gerenciamento eficiente de memória."""
    chunks = [text[i:i+4096] for i in range(0, len(text), 3072)]  # Sobreposição de 1024
    results = []
    
    for chunk in chunks:
        result = optimized_model.generate_text(chunk, max_length=512)
        results.append(result)
        
        # Limpar caches depois de cada chunk
        memory_optimizer.clear_kv_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return "\n".join(results)
```

### Erros de Serialização e Desserialização

**Problema**: Falhas ao salvar ou carregar modelos com componentes avançados.

**Sintomas**:
- Erros de pickle ao salvar
- Erros ao carregar componentes personalizados
- Modelos carregados sem componentes MoE ou State-Space

**Solução**:
```python
from lunagpt import LunaModel
from lunagpt.utils import SerializationHelper

# Carregar modelo
model = LunaModel.from_pretrained("./models/luna-2.6")

# Objeto auxiliar para serialização
serializer = SerializationHelper()

# Salvar com tratamento especial para componentes avançados
save_path = "./models/luna-2.6-custom"
serializer.save_model_with_components(
    model,
    save_path,
    special_components=["moe_blocks", "state_space_layers", "hypernet"]
)

# Restaurar modelo com componentes especiais
restored_model = serializer.load_model_with_components(save_path)

# Verificar se componentes foram preservados
assert hasattr(restored_model, "moe_blocks")
assert hasattr(restored_model, "state_space_layers")
```

### Problemas de Treinamento

**Problema**: Instabilidade durante treinamento, especialmente com componentes avançados.

**Sintomas**:
- Explosão ou desaparecimento de gradientes
- Perda divergente ou estagnada
- Desempenho inconsistente após treinamento

**Solução**:
```python
from lunagpt.training import StableTrainer
from lunagpt import LunaModel

# Carregar modelo para treinamento estável
model = LunaModel.from_pretrained("./models/luna-2.6")

# Inicializar trainer com estabilizadores
trainer = StableTrainer(
    model=model,
    # Estabilizadores ativados
    use_gradient_clipping=True,
    gradient_clip_value=1.0,
    use_weight_decay=True,
    weight_decay=0.01,
    use_skip_connection_tuning=True,
    use_layer_normalization=True,
    enable_ema=True,  # Exponential Moving Average dos pesos
    ema_decay=0.999,
    warmup_steps=1000
)

# Configurar monitoramento para detectar problemas
trainer.set_monitors([
    "gradient_norm",
    "parameter_norm",
    "loss_spikes",
    "activation_statistics"
])

# Configurar callbacks adaptativos
trainer.add_adaptive_callbacks([
    "lr_plateau_reducer",
    "gradient_clipper_adjuster",
    "early_divergence_stopper"
])

# Treinar com estabilização
training_result = trainer.train(
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    batch_size=8,
    epochs=5
)

# Verificar qualidade do treinamento
stability_report = training_result["stability_report"]
if stability_report["is_stable"]:
    print("Treinamento estável concluído com sucesso.")
    model.save("./models/luna-2.6-finetuned")
else:
    print(f"Problemas detectados: {stability_report['issues']}")
    print("Aplicando correções...")
    fixed_model = trainer.apply_stability_fixes(stability_report)
    fixed_model.save("./models/luna-2.6-finetuned-fixed")
```

### Incompatibilidade de Tokenização

**Problema**: Problemas na tokenização específica para português.

**Sintomas**:
- Tokenização incorreta de acentos e caracteres especiais
- Perda de informação em contrações portuguesas
- Problemas na manipulação de conjugações verbais

**Solução**:
```python
from lunagpt.tokenizer import LunaTokenizer, PortugueseTokenizerFixer

# Carregar tokenizador
config = {"tokenizer": {"vocab_path": "./models/luna-2.6/tokenizer/vocab.json", "merges_path": "./models/luna-2.6/tokenizer/merges.txt"}}
tokenizer = LunaTokenizer(config)

# Verificar problemas com texto português
sample_text = "À noite, João contou-nos que não queria ir à festa."
tokens = tokenizer.encode(sample_text)
decoded = tokenizer.decode(tokens)

# Verificar perda de informação
if sample_text != decoded:
    print(f"Perda na tokenização detectada:\nOriginal: {sample_text}\nReconstruído: {decoded}")
    
    # Aplicar correções para português
    fixer = PortugueseTokenizerFixer(tokenizer)
    
    # Corrigir tratamento de contrações
    fixer.add_contraction_rules()
    
    # Corrigir tratamento de pronomes clíticos
    fixer.add_clitic_rules()
    
    # Corrigir tratamento de acentuação
    fixer.preserve_accentuation()
    
    # Testar novamente
    tokens_fixed = tokenizer.encode(sample_text)
    decoded_fixed = tokenizer.decode(tokens_fixed)
    
    print(f"Após correções:\nReconstruído: {decoded_fixed}")
    print(f"Preservação correta: {sample_text == decoded_fixed}")
    
    # Salvar tokenizador melhorado
    tokenizer.save_pretrained("./models/luna-2.6/tokenizer-enhanced/")
```

## Guia para Desenvolvedores

### Estendendo o LunaGPT

O LunaGPT foi projetado para ser extensível, permitindo que desenvolvedores adicionem novas funcionalidades:

Similar code found with 1 license type