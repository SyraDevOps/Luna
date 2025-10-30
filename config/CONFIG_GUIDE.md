# Sistema de Configuração Luna - Guia Completo

## Visão Geral

O Luna GPT possui um sistema de configuração completo e flexível que permite ajustar todos os aspectos do sistema através de:
1. Arquivos de configuração JSON
2. Variáveis de ambiente
3. Argumentos da linha de comando (CLI)

A ordem de precedência é: **CLI > Variáveis de Ambiente > Arquivo JSON > Padrões**

## Configuração via Linha de Comando (CLI)

### Parâmetros Globais

```bash
python main.py --config config/custom.json --log-level DEBUG --no-timeout [comando]
```

Opções globais:
- `--config`: Caminho para arquivo de configuração JSON personalizado
- `--log-level`: Nível de logging (DEBUG, INFO, WARNING, ERROR)
- `--no-timeout`: Desabilitar todos os timeouts do sistema

### Criar Modelo

```bash
python main.py create --model meu_modelo \
  --vocab-size 32000 \
  --hidden-size 768 \
  --num-layers 12 \
  --use-moe \
  --num-experts 8 \
  --use-state-space \
  --use-hypernet \
  --train-data "data/train/*.txt"
```

Parâmetros:
- `--model`: Nome do modelo (obrigatório)
- `--vocab-size`: Tamanho do vocabulário
- `--hidden-size`: Dimensão das camadas ocultas
- `--num-layers`: Número de camadas transformer
- `--use-moe`: Habilitar Mixture of Experts
- `--num-experts`: Número de especialistas no MoE
- `--use-state-space`: Habilitar State-Space Layers
- `--use-hypernet`: Habilitar HyperNetworks
- `--train-data`: Padrão de arquivos para dados iniciais

### Treinar Modelo

```bash
python main.py train --model meu_modelo \
  --train-data "data/train/*.json" \
  --valid-data "data/valid/*.json" \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --gradient-accumulation 2 \
  --max-grad-norm 1.0 \
  --use-wandb \
  --use-curriculum \
  --use-automl \
  --automl-trials 20 \
  --fp16 \
  --gradient-checkpointing
```

Parâmetros:
- `--model`: Nome do modelo (obrigatório)
- `--train-data`: Padrão de arquivos de treinamento
- `--valid-data`: Padrão de arquivos de validação
- `--epochs`: Número de épocas de treinamento
- `--batch-size`: Tamanho do batch
- `--learning-rate`: Taxa de aprendizado
- `--weight-decay`: Decaimento de peso (L2 regularization)
- `--warmup-ratio`: Proporção de passos para warmup
- `--gradient-accumulation`: Passos para acumular gradiente
- `--max-grad-norm`: Norma máxima para gradient clipping
- `--use-wandb`: Ativar Weights & Biases para tracking
- `--use-curriculum`: Usar curriculum learning
- `--use-automl`: Usar AutoML para otimização de hiperparâmetros
- `--automl-trials`: Número de trials do AutoML
- `--fp16`: Usar precisão mista (FP16)
- `--gradient-checkpointing`: Ativar gradient checkpointing para economizar memória

### Chat Interativo (Terminal)

```bash
python main.py chat --model meu_modelo \
  --persona tecnico \
  --use-rag \
  --use-proactive \
  --max-history 20 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-tokens 512
```

Parâmetros:
- `--model`: Nome do modelo (obrigatório)
- `--persona`: Persona a usar (casual, tecnico, formal, amigavel, profissional)
- `--use-rag`: Ativar Retrieval-Augmented Generation
- `--use-proactive`: Ativar mensagens proativas
- `--max-history`: Máximo de mensagens no histórico
- `--temperature`: Temperatura de geração (0.0-2.0)
- `--top-p`: Top-p sampling (nucleus sampling)
- `--max-tokens`: Máximo de tokens a gerar

### Interface Web (HTML)

```bash
python main.py web --model meu_modelo \
  --host 0.0.0.0 \
  --port 5000 \
  --debug
```

Parâmetros:
- `--model`: Nome do modelo (obrigatório)
- `--host`: Host do servidor (padrão: 0.0.0.0)
- `--port`: Porta do servidor (padrão: 5000)
- `--debug`: Ativar modo debug do Flask

### Refinamento com Feedback

```bash
python main.py refine --model meu_modelo \
  --min-samples 10 \
  --quality-threshold 4
```

Parâmetros:
- `--model`: Nome do modelo (obrigatório)
- `--min-samples`: Mínimo de amostras para iniciar refinamento
- `--quality-threshold`: Threshold mínimo de qualidade (1-5)

### Gerenciar Configuração

```bash
# Mostrar configuração atual
python main.py config --action show

# Salvar configuração atual
python main.py config --action save --output config/minha_config.json

# Validar configuração
python main.py config --action validate
```

## Configuração via Arquivo JSON

Crie um arquivo JSON com as configurações desejadas:

```json
{
  "model": {
    "hidden_size": 1024,
    "num_hidden_layers": 16,
    "use_moe": true,
    "num_experts": 16
  },
  "training": {
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 8,
    "num_train_epochs": 10
  },
  "rag": {
    "use_rag": true,
    "top_k_retrieval": 10
  }
}
```

Use o arquivo:
```bash
python main.py --config config/minha_config.json train --model meu_modelo
```

## Configuração via Variáveis de Ambiente

Configure variáveis de ambiente para sobrescrever configurações:

```bash
export LUNA_LEARNING_RATE=3e-5
export LUNA_BATCH_SIZE=8
export LUNA_EPOCHS=10
export LUNA_USE_WANDB=true
export LUNA_MODEL_SIZE=1024
export LUNA_USE_RAG=true
export LUNA_FEEDBACK_THRESHOLD=4

python main.py train --model meu_modelo
```

Variáveis disponíveis:
- `LUNA_LEARNING_RATE`: Taxa de aprendizado
- `LUNA_BATCH_SIZE`: Tamanho do batch
- `LUNA_EPOCHS`: Número de épocas
- `LUNA_USE_WANDB`: Usar W&B (true/false)
- `LUNA_MODEL_SIZE`: Tamanho do modelo (hidden_size)
- `LUNA_USE_RAG`: Usar RAG (true/false)
- `LUNA_FEEDBACK_THRESHOLD`: Threshold de qualidade do feedback
- `LUNA_NO_TIMEOUT`: Desabilitar timeouts (true/false)

## Seções de Configuração

### 1. Tokenizer
- `vocab_size`: Tamanho do vocabulário
- `max_length`: Comprimento máximo de sequência
- `min_frequency`: Frequência mínima para incluir token
- `special_tokens`: Tokens especiais (pad, unk, bos, eos, etc.)

### 2. Model
- `model_name`: Nome do modelo
- `hidden_size`: Dimensão das representações ocultas
- `num_hidden_layers`: Número de camadas transformer
- `num_attention_heads`: Número de cabeças de atenção
- `use_moe`: Ativar Mixture of Experts
- `num_experts`: Número de especialistas
- `use_state_space`: Ativar State-Space Layers
- `use_hypernet`: Ativar HyperNetworks
- `use_growing_network`: Ativar GrowingNetwork

### 3. Training
- `num_train_epochs`: Número de épocas
- `per_device_train_batch_size`: Batch size por dispositivo
- `learning_rate`: Taxa de aprendizado
- `weight_decay`: Peso de decaimento
- `warmup_ratio`: Proporção de warmup
- `use_wandb`: Usar Weights & Biases
- `use_curriculum`: Usar curriculum learning
- `fp16`: Usar precisão mista

### 4. RAG (Retrieval-Augmented Generation)
- `use_rag`: Ativar sistema RAG
- `index_path`: Caminho para índice de documentos
- `embedding_model`: Modelo de embeddings
- `top_k_retrieval`: Número de documentos a recuperar
- `similarity_threshold`: Threshold de similaridade

### 5. Feedback
- `feedback_file`: Arquivo de feedback
- `quality_threshold`: Threshold de qualidade (1-5)
- `min_samples_for_update`: Mínimo de amostras para refinamento
- `collect_user_feedback`: Coletar feedback do usuário

### 6. Memory
- `memory_file`: Arquivo de memória persistente
- `max_memory_entries`: Máximo de entradas de memória
- `use_semantic_memory`: Usar memória semântica

### 7. Persona
- `default_persona`: Persona padrão
- `available_personas`: Lista de personas disponíveis

### 8. Optimization
- `use_automl`: Usar AutoML
- `automl_trials`: Número de trials do AutoML
- `use_dynamic_batching`: Usar batching dinâmico
- `auto_device_map`: Mapeamento automático de dispositivos

## Exemplos de Uso

### Sistema Elite de Produção (1000/1000)

```bash
# Criar modelo elite com todas as funcionalidades
python main.py create --model luna_elite \
  --vocab-size 50000 \
  --hidden-size 1024 \
  --num-layers 24 \
  --use-moe \
  --num-experts 16 \
  --use-state-space \
  --use-hypernet \
  --train-data "data/train/**/*.json"

# Treinar com configuração ótima
python main.py train --model luna_elite \
  --train-data "data/train/**/*.json" \
  --valid-data "data/valid/**/*.json" \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --use-wandb \
  --use-curriculum \
  --use-automl \
  --automl-trials 50 \
  --fp16 \
  --gradient-checkpointing \
  --no-timeout

# Iniciar chat com todas as funcionalidades
python main.py chat --model luna_elite \
  --persona profissional \
  --use-rag \
  --use-proactive \
  --max-history 50 \
  --temperature 0.7 \
  --max-tokens 1024

# Interface web para produção
python main.py web --model luna_elite \
  --host 0.0.0.0 \
  --port 8080
```

### Sistema de Desenvolvimento Rápido

```bash
# Modelo leve para desenvolvimento
python main.py create --model luna_dev \
  --vocab-size 10000 \
  --hidden-size 256 \
  --num-layers 4 \
  --train-data "data/train/sample.txt"

# Treinamento rápido
python main.py train --model luna_dev \
  --epochs 1 \
  --batch-size 2 \
  --learning-rate 5e-5

# Chat simples
python main.py chat --model luna_dev
```

## Dicas de Otimização

### Para Hardware Limitado
- Reduza `hidden_size` e `num_hidden_layers`
- Use `--gradient-checkpointing`
- Reduza `batch_size` e aumente `gradient_accumulation`
- Desative componentes pesados: sem MoE, State-Space, HyperNet

### Para Máxima Performance
- Aumente `hidden_size`, `num_hidden_layers`, `num_experts`
- Use `--fp16` ou FP16 misto
- Ative todos os componentes: MoE, State-Space, HyperNet
- Use `--use-automl` para encontrar hiperparâmetros ótimos
- Configure `--no-timeout` para processos longos

### Para Máxima Qualidade
- Use `--use-curriculum` para treinamento progressivo
- Ative `--use-rag` para respostas fundamentadas
- Use `--use-proactive` para interações inteligentes
- Configure feedback contínuo com `refine`
- Treine por mais épocas com learning rate menor
