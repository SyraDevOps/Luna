# 🚀 Luna GPT - Sistema Elite de Conversação AI (1000/1000)

## 🌟 Visão Geral

Luna GPT é um sistema avançado de diálogo neural adaptativo, projetado para ser um assistente AI de classe mundial com integração completa de:
- ✅ **Mixture of Experts (MoE)** - Especialização dinâmica
- ✅ **State-Space Layers** - Processamento eficiente de contexto longo
- ✅ **HyperNetworks** - Adaptação contextual dinâmica
- ✅ **Growing Networks** - Expansão neural orgânica
- ✅ **RAG (Retrieval-Augmented Generation)** - Respostas fundamentadas
- ✅ **Sistema de Feedback** - Aprendizado contínuo
- ✅ **Mensagens Proativas** - Interação inteligente
- ✅ **Chat Terminal e Web** - Interfaces múltiplas
- ✅ **Configuração Total via CLI** - Controle completo
- ✅ **AutoML** - Otimização automática de hiperparâmetros
- ✅ **Curriculum Learning** - Treinamento progressivo
- ✅ **Sem Timeouts** - Operação ininterrupta configurável

## 🎯 Características Elite

### 1. Arquitetura Híbrida Avançada
- **Transformer Base**: Modelagem de contexto local de alta qualidade
- **State-Space Models**: Processamento linear eficiente para sequências longas
- **Mixture of Experts**: Aumento de capacidade com eficiência computacional
- **HyperNetworks**: Geração dinâmica de parâmetros adaptativos
- **Growing Networks**: Crescimento orgânico conforme necessidade

### 2. Processamento de Linguagem Natural Elite
- **NLTK**: Tokenização e análise sintática
- **Stanza**: Análise linguística avançada para português
- **Sentence Transformers**: Embeddings semânticos de última geração
- **Tokenizador Customizado**: Otimizado especificamente para português

### 3. Sistema de Conhecimento Integrado
- **RAG Adaptativo**: Recuperação de documentos relevantes
- **Indexação Vetorial**: Busca semântica eficiente com FAISS
- **Memória Semântica**: Contexto persistente e recuperação inteligente
- **Base de Conhecimento Dinâmica**: Atualização contínua

### 4. Múltiplas Interfaces
- **Terminal/CLI**: Chat interativo via linha de comando
- **Web/HTML**: Interface web moderna e responsiva
- **API REST**: Integração com outros sistemas
- **Configuração via CLI**: Controle total de todos os parâmetros

## 📦 Instalação

### Requisitos
- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM (8GB+ recomendado)
- GPU NVIDIA com CUDA (opcional, mas recomendado)

### Instalação Rápida

```bash
# Clonar repositório
git clone https://github.com/SyraDevOps/Luna.git
cd Luna

# Instalar dependências
pip install -r requirements.txt

# Baixar recursos NLTK (automático no primeiro uso)
python -c "import nltk; nltk.download('punkt')"
```

## 🚀 Início Rápido

### 1. Criar Modelo Elite

```bash
# Modelo de produção completo (recomendado para uso sério)
python main.py create --model luna_elite \
  --vocab-size 50000 \
  --hidden-size 1024 \
  --num-layers 24 \
  --use-moe \
  --num-experts 16 \
  --use-state-space \
  --use-hypernet \
  --train-data "data/train/**/*.json"
```

### 2. Treinar com Otimização Automática

```bash
# Treinamento completo com AutoML e todas as otimizações
python main.py train --model luna_elite \
  --train-data "data/train/**/*.json" \
  --valid-data "data/valid/**/*.json" \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --use-wandb \
  --use-curriculum \
  --use-automl \
  --automl-trials 50 \
  --fp16 \
  --gradient-checkpointing \
  --no-timeout
```

### 3. Chat Interativo (Terminal)

```bash
# Chat completo com todas as funcionalidades
python main.py chat --model luna_elite \
  --persona profissional \
  --use-rag \
  --use-proactive \
  --max-history 50 \
  --temperature 0.7 \
  --max-tokens 1024
```

### 4. Interface Web (HTML)

```bash
# Iniciar servidor web
python main.py web --model luna_elite \
  --host 0.0.0.0 \
  --port 8080

# Acessar no navegador: http://localhost:8080
```

## 📚 Documentação Completa

### Guias Disponíveis

1. **[CONFIG_GUIDE.md](config/CONFIG_GUIDE.md)** - Guia completo de configuração
   - Todos os parâmetros CLI documentados
   - Configuração via JSON e variáveis de ambiente
   - Exemplos para diferentes cenários
   - Dicas de otimização para hardware variado

2. **[README.md](README.md)** - Documentação principal do sistema
   - Arquitetura detalhada
   - Fundamentos teóricos
   - Componentes técnicos
   - Guias de uso avançado

## 🎮 Comandos Principais

### Gerenciamento de Modelos

```bash
# Criar novo modelo
python main.py create --model NOME [opções]

# Treinar modelo existente
python main.py train --model NOME [opções]

# Refinar com feedback
python main.py refine --model NOME
```

### Interfaces de Chat

```bash
# Chat terminal
python main.py chat --model NOME [opções]

# Interface web
python main.py web --model NOME --port 8080
```

### Configuração e Utilitários

```bash
# Ver configuração atual
python main.py config --action show

# Salvar configuração
python main.py config --action save --output config/custom.json

# Validar configuração
python main.py config --action validate

# Executar testes
python main.py test
```

## ⚙️ Configuração Avançada

### Arquivo de Configuração JSON

```json
{
  "model": {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "use_moe": true,
    "num_experts": 16,
    "use_state_space": true,
    "use_hypernet": true
  },
  "training": {
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "use_wandb": true,
    "use_curriculum": true
  },
  "rag": {
    "use_rag": true,
    "top_k_retrieval": 10
  }
}
```

Usar configuração:
```bash
python main.py --config config/custom.json train --model NOME
```

### Variáveis de Ambiente

```bash
export LUNA_LEARNING_RATE=2e-5
export LUNA_BATCH_SIZE=16
export LUNA_USE_WANDB=true
export LUNA_NO_TIMEOUT=true

python main.py train --model NOME
```

## 🏗️ Arquitetura do Sistema

```
Luna GPT
├── Interface de Usuário (CLI, Web, API)
├── Sistema de Chat
│   ├── Gerenciamento de Sessão
│   ├── Formatação de Mensagens
│   └── Sistema Proativo
├── Núcleo do Modelo
│   ├── Transformer Base
│   ├── Mixture of Experts (MoE)
│   ├── State-Space Layers
│   ├── HyperNetworks
│   └── Growing Networks
├── Sistema RAG
│   ├── Indexação de Documentos
│   ├── Busca Semântica
│   └── Aumento de Contexto
├── Sistema de Treinamento
│   ├── Supervised Learning
│   ├── Curriculum Learning
│   ├── Feedback Learning
│   └── AutoML
└── Componentes de Suporte
    ├── Tokenizador Customizado
    ├── Sistema de Memória
    ├── Sistema de Feedback
    └── Gerenciamento de Personas
```

## 🔧 Otimização para Diferentes Hardware

### Hardware Limitado (4GB RAM, sem GPU)

```bash
python main.py create --model luna_lite \
  --vocab-size 10000 \
  --hidden-size 256 \
  --num-layers 4 \
  --train-data "data/train/sample.txt"

python main.py train --model luna_lite \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --epochs 3
```

### Hardware Moderado (8GB RAM, GPU 4GB)

```bash
python main.py create --model luna_medium \
  --vocab-size 32000 \
  --hidden-size 512 \
  --num-layers 8 \
  --use-moe \
  --num-experts 4

python main.py train --model luna_medium \
  --batch-size 8 \
  --fp16 \
  --gradient-checkpointing
```

### Hardware Potente (16GB+ RAM, GPU 8GB+)

```bash
python main.py create --model luna_elite \
  --vocab-size 50000 \
  --hidden-size 1024 \
  --num-layers 24 \
  --use-moe \
  --num-experts 16 \
  --use-state-space \
  --use-hypernet

python main.py train --model luna_elite \
  --batch-size 16 \
  --use-automl \
  --automl-trials 50 \
  --fp16 \
  --no-timeout
```

## 📊 Monitoramento e Logging

### Níveis de Log

```bash
# Debug completo
python main.py --log-level DEBUG comando

# Apenas informações importantes
python main.py --log-level INFO comando

# Apenas warnings e erros
python main.py --log-level WARNING comando
```

### Integração com Weights & Biases

```bash
# Ativar tracking W&B
python main.py train --model NOME --use-wandb

# Ver métricas no dashboard W&B
wandb login
# Acesse wandb.ai para visualizar
```

## 🧪 Testes

```bash
# Executar todos os testes
python main.py test

# Executar testes específicos
python -m pytest src/tests/test_specific.py -v
```

## 🔍 Resolução de Problemas

### Problema: Timeout durante treinamento

**Solução**: Use `--no-timeout`
```bash
python main.py train --model NOME --no-timeout
```

### Problema: Falta de memória

**Soluções**:
1. Reduzir batch size: `--batch-size 2`
2. Usar gradient checkpointing: `--gradient-checkpointing`
3. Usar FP16: `--fp16`
4. Aumentar gradient accumulation: `--gradient-accumulation 8`

### Problema: Modelo não treina bem

**Soluções**:
1. Usar curriculum learning: `--use-curriculum`
2. Usar AutoML: `--use-automl --automl-trials 20`
3. Ajustar learning rate: `--learning-rate 1e-5`
4. Verificar dados de treinamento

## 🤝 Contribuindo

Contribuições são bem-vindas! Veja [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes.

## 📄 Licença

Este projeto está licenciado sob a licença especificada no arquivo [LICENSE](LICENSE).

## 🙏 Agradecimentos

- Comunidade PyTorch
- HuggingFace Transformers
- Sentence Transformers
- FAISS
- NLTK e Stanza teams

## 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/SyraDevOps/Luna/issues)
- **Documentação**: Ver arquivos de documentação no repositório
- **Exemplos**: Ver pasta `examples/` (em desenvolvimento)

---

**Luna GPT - Conversação AI de Próxima Geração 🚀**

*Desenvolvido com dedicação para ser um sistema 1000/1000 como proposto* ✨
