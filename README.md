# Luna

![Luna](img/Luna.png)


 ```markdown
# LunaGPT - Documentação Técnica e Guia de Uso

## Visão Geral
LunaGPT é um modelo de linguagem avançado com arquitetura híbrida, combinando técnicas de **MoE (Mixture of Experts)**, **RAG (Retrieval-Augmented Generation)** e **treinamento contínuo baseado em feedback**. Projetado para operar em hardware modesto, inclui otimizações como poda estrutural (30%) e quantização de 4 bits.

---

## Funcionalidades Principais
- **Memória de Contexto Persistente**: Armazena histórico em SQLite para coerência em conversas longas
- **Inferência com RAG**: Integração com FAISS para recuperação de dados em tempo real
- **Feedback Contínuo**: Sistema de avaliação Likert/NPS com atualização automática do modelo
- **Personalização**: 3 personas pré-definidas (técnico, casual, formal)
- **Hardware Modesto**: Otimizado para GPUs com 8GB+ de VRAM

---

## Requisitos de Sistema
```bash
# Dependências principais
Python 3.8+
PyTorch 2.0+
CUDA 11.8+ (opcional)
Transformers 4.35+
Faiss-CPU/GPU
SQLite3
```

---

## Instalação
```bash
# Clone o repositório
git clone https://github.com/syra-team/luna-gpt
cd luna-gpt

# Instale dependências
pip install -r requirements.txt

# Configure CUDA (opcional)
export CUDA_VISIBLE_DEVICES=0
```

---

## Uso Básico

### 1. Treinamento Inicial
```bash
# Formato de dados suportados: CSV, JSON, PDF, TXT
python luna.py create --name "meu_modelo" --train_data "data/*.csv"

# Treinamento com cross-validation
python luna.py train --model "meu_modelo" --epochs 10 --train_data "data/*.json"
```

### 2. Interface de Chat
```bash
python luna.py chat --model "meu_modelo" --persona "tecnico"

# Exemplo de interação
Você: Explique a teoria da relatividade
Luna: [TÉCNICO] A relatividade geral descreve a gravidade como...
```

### 3. Sistema de Feedback
```python
# Após cada resposta, avalie:
Avalie a resposta (1-5): 5
Avalie a usabilidade (1-7): 7
Net Promoter Score (-100 a 100): 90
```

---

## Configuração Avançada

### Parâmetros do Modelo (config.py)
```python
# Quantização dinâmica
quantization = QuantizationConfig(
    enabled=True,
    method="awq",  # Alternativas: gptq, bitsandbytes
    bits=4
)

# RAG Config
rag = RAGConfig(
    enabled=True,
    index_path="faiss_index.idx",
    dim=768
)
```

### Uso de Personas
| Persona  | Temperatura | Uso Ideal                |
|----------|-------------|--------------------------|
| técnico  | 0.2         | Respostas precisas       |
| casual   | 0.7         | Conversas informais      |
| formal   | 0.3         | Comunicação empresarial  |

---

## Pipeline de Treinamento

### 1. Preparação de Dados
```python
# Exemplo de CSV válido
pergunta,resposta
"Qual a capital da França?","Paris"
"Explique IA","Inteligência Artificial..." 

# Estrutura JSON válida
[
    {"pergunta": "Pergunta 1", "resposta": "Resposta 1"},
    {"sender": "Usuário", "message": "Mensagem 2"}
]
```

### 2. Treinamento Multifásico
```bash
# Fase 1: Treinamento supervisionado
python luna.py train --model "meu_modelo" --train_data "data/*.csv"

# Fase 2: Refinamento com RLHF
python luna.py refine --model "meu_modelo"
```

---

## Métricas de Avaliação
| Métrica       | Comando de Cálculo                          |
|---------------|--------------------------------------------|
| Perplexidade  | `compute_perplexity(model, tokenizer, dados)` |
| BLEU          | `compute_bleu(predições, referências)`     |

---

## Atualizações Contínuas
```python
# Adicione novos dados ao diretório /data
# Execute atualização automática:
python luna.py train --model "meu_modelo" --train_data "novos_dados/*.json"
```

---

## Limitações Conhecidas
- Requer pelo menos 8GB de RAM para inferência
- Tempo de resposta aumenta 15% com RAG habilitado
- Suporte limitado a idiomas não latinos

---

## Exemplo de Código para Integração
```python
from luna import LunaChat

chat = LunaChat("meu_modelo", persona="casual")
resposta = chat.generate("Como funciona a Lua?")
print(resposta)  # Saída: "A Lua orbita a Terra..."
```

---

## Referências
 Modelo de e-commerce Luna  
 Regras de negócio baseadas em ciclos lunares  
 Arquitetura MoE vs Discriminativa  
 Módulo de segurança HSM Luna PCIe
```
