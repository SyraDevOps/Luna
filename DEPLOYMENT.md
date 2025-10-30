# 🚀 Luna GPT - Guia de Implantação e Operações

## Índice

1. [Preparação do Ambiente](#preparação-do-ambiente)
2. [Instalação e Configuração](#instalação-e-configuração)
3. [Verificação do Sistema](#verificação-do-sistema)
4. [Implantação](#implantação)
5. [Operação e Manutenção](#operação-e-manutenção)
6. [Monitoramento](#monitoramento)
7. [Troubleshooting](#troubleshooting)
8. [Backup e Recuperação](#backup-e-recuperação)

---

## Preparação do Ambiente

### Requisitos Mínimos

**Para Desenvolvimento:**
- Python 3.8+
- 4GB RAM
- 10GB espaço em disco
- CPU multi-core

**Para Produção:**
- Python 3.9+
- 16GB+ RAM
- 50GB+ espaço em disco
- GPU NVIDIA com 8GB+ VRAM (recomendado)
- CUDA 11.8+

### Preparação do Sistema

```bash
# Atualizar sistema (Ubuntu/Debian)
sudo apt update && sudo apt upgrade -y

# Instalar dependências do sistema
sudo apt install -y python3-dev python3-pip build-essential
sudo apt install -y git wget curl

# Instalar CUDA (opcional, para GPU)
# Seguir guia oficial: https://developer.nvidia.com/cuda-downloads
```

---

## Instalação e Configuração

### 1. Clonar Repositório

```bash
cd /opt  # ou diretório de sua preferência
git clone https://github.com/SyraDevOps/Luna.git
cd Luna
```

### 2. Criar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instalar Dependências

```bash
# Instalar pacotes principais
pip install --upgrade pip
pip install -r requirements.txt

# Baixar recursos NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 4. Configuração Inicial

```bash
# Criar estrutura de diretórios
python -c "
import os
for d in ['data/train', 'data/valid', 'data/rag_index', 'models', 'logs', 'temp', 'config']:
    os.makedirs(d, exist_ok=True)
"

# Copiar configuração padrão
cp config/luna_default_config.json config/production.json

# Editar configuração para produção
nano config/production.json
```

### 5. Configurar Variáveis de Ambiente

```bash
# Criar arquivo .env
cat > .env << EOF
LUNA_USE_WANDB=true
LUNA_NO_TIMEOUT=true
LUNA_MODEL_SIZE=1024
LUNA_USE_RAG=true
WANDB_API_KEY=seu_api_key_aqui
EOF

# Carregar variáveis
source .env
```

---

## Verificação do Sistema

### Verificação Rápida

```bash
# Verificar status geral
python system_status.py

# Verificar integração de componentes
python verify_integration.py

# Executar testes
python main.py test
```

### Checklist de Verificação

- [ ] Python 3.8+ instalado
- [ ] Todas as dependências instaladas
- [ ] GPU detectada (se aplicável)
- [ ] Estrutura de diretórios criada
- [ ] Configuração validada
- [ ] Testes básicos passando
- [ ] Variáveis de ambiente configuradas

---

## Implantação

### Ambiente de Desenvolvimento

```bash
# 1. Criar modelo de desenvolvimento
python main.py create --model luna_dev \
  --vocab-size 10000 \
  --hidden-size 256 \
  --num-layers 4 \
  --train-data "data/train/*.txt"

# 2. Treinar rapidamente
python main.py train --model luna_dev \
  --epochs 1 \
  --batch-size 2

# 3. Testar chat
python main.py chat --model luna_dev
```

### Ambiente de Produção

#### Opção 1: Chat Terminal

```bash
# 1. Criar modelo elite
python main.py --config config/production.json create --model luna_prod \
  --vocab-size 50000 \
  --hidden-size 1024 \
  --num-layers 24 \
  --use-moe \
  --num-experts 16 \
  --use-state-space \
  --use-hypernet \
  --train-data "data/train/**/*.json"

# 2. Treinar com AutoML
python main.py --config config/production.json --no-timeout \
  train --model luna_prod \
  --train-data "data/train/**/*.json" \
  --valid-data "data/valid/**/*.json" \
  --epochs 20 \
  --batch-size 16 \
  --use-wandb \
  --use-curriculum \
  --use-automl \
  --automl-trials 100 \
  --fp16 \
  --gradient-checkpointing

# 3. Iniciar chat em produção
python main.py chat --model luna_prod \
  --persona profissional \
  --use-rag \
  --use-proactive \
  --max-history 100
```

#### Opção 2: Interface Web

```bash
# 1. Usar modelo treinado
# (seguir passos 1-2 da Opção 1)

# 2. Iniciar servidor web
python main.py web --model luna_prod \
  --host 0.0.0.0 \
  --port 8080
```

#### Opção 3: Servidor Produção com Gunicorn

```bash
# Instalar gunicorn
pip install gunicorn

# Criar arquivo wsgi.py
cat > wsgi.py << 'EOF'
from src.web.app import create_app
from src.config.config import Config

config = Config('config/production.json')
app = create_app('luna_prod', config)

if __name__ == "__main__":
    app.run()
EOF

# Iniciar com gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 wsgi:app --timeout 300
```

#### Opção 4: Docker (Recomendado para Produção)

```bash
# Criar Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicação
COPY . .

# Baixar recursos NLTK
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Expor porta
EXPOSE 8080

# Comando padrão
CMD ["python", "main.py", "web", "--model", "luna_prod", "--host", "0.0.0.0", "--port", "8080"]
EOF

# Build e executar
docker build -t luna-gpt:latest .
docker run -d -p 8080:8080 -v $(pwd)/models:/app/models luna-gpt:latest
```

---

## Operação e Manutenção

### Rotinas Diárias

```bash
# Verificar status do sistema
python system_status.py

# Verificar logs
tail -f logs/luna_*.log

# Verificar uso de recursos
htop  # ou top
nvidia-smi  # para GPU
```

### Rotinas Semanais

```bash
# Backup de modelos
tar -czf backup_models_$(date +%Y%m%d).tar.gz models/

# Limpar logs antigos (mantém últimos 30 dias)
find logs/ -name "*.log" -mtime +30 -delete

# Refinar modelos com feedback
python main.py refine --model luna_prod \
  --min-samples 100 \
  --quality-threshold 4
```

### Rotinas Mensais

```bash
# Retreinar com novos dados
python main.py train --model luna_prod \
  --train-data "data/train/**/*.json" \
  --epochs 5 \
  --use-curriculum

# Atualizar dependências
pip install --upgrade -r requirements.txt

# Executar testes completos
python main.py test
python verify_integration.py
```

---

## Monitoramento

### Monitoramento Local

```bash
# Verificar saúde da API web
curl http://localhost:8080/api/health

# Monitorar logs em tempo real
tail -f logs/luna_*.log | grep ERROR

# Verificar uso de GPU
watch -n 1 nvidia-smi
```

### Monitoramento com Weights & Biases

```bash
# Configurar W&B
wandb login

# Treinar com tracking
python main.py train --model luna_prod --use-wandb

# Ver métricas no dashboard
# Acessar: https://wandb.ai
```

### Métricas Importantes

**Sistema:**
- Uso de CPU/GPU
- Uso de memória
- Temperatura GPU
- Espaço em disco

**Aplicação:**
- Taxa de requisições
- Tempo de resposta
- Taxa de erro
- Qualidade das respostas

**Modelo:**
- Loss de treinamento
- Perplexidade
- Acurácia
- Feedback do usuário

---

## Troubleshooting

### Problema: Falta de Memória

**Sintomas:**
```
RuntimeError: CUDA out of memory
```

**Soluções:**
```bash
# 1. Reduzir batch size
python main.py train --model NOME --batch-size 2

# 2. Usar gradient checkpointing
python main.py train --model NOME --gradient-checkpointing

# 3. Usar FP16
python main.py train --model NOME --fp16

# 4. Aumentar gradient accumulation
python main.py train --model NOME --gradient-accumulation 8
```

### Problema: Timeout durante Treinamento

**Sintomas:**
```
TimeoutError: Operation timed out
```

**Soluções:**
```bash
# Desabilitar timeouts
python main.py --no-timeout train --model NOME
```

### Problema: Modelo não Carrega

**Sintomas:**
```
FileNotFoundError: Model not found
```

**Soluções:**
```bash
# 1. Verificar se modelo existe
ls -la models/

# 2. Verificar permissões
chmod -R 755 models/

# 3. Recriar modelo
python main.py create --model NOME
```

### Problema: Dependências Faltando

**Sintomas:**
```
ModuleNotFoundError: No module named 'X'
```

**Soluções:**
```bash
# Reinstalar dependências
pip install -r requirements.txt

# Verificar instalação
python verify_integration.py
```

---

## Backup e Recuperação

### Estratégia de Backup

**Diário:**
- Logs do dia
- Feedback do usuário
- Configuração atual

**Semanal:**
- Modelos treinados
- Índices RAG
- Dados de memória

**Mensal:**
- Backup completo do sistema
- Dados de treinamento
- Histórico de versões

### Scripts de Backup

```bash
# Backup completo
#!/bin/bash
BACKUP_DIR="/backup/luna/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

tar -czf $BACKUP_DIR/models.tar.gz models/
tar -czf $BACKUP_DIR/data.tar.gz data/
tar -czf $BACKUP_DIR/config.tar.gz config/
tar -czf $BACKUP_DIR/logs.tar.gz logs/

echo "Backup completo em $BACKUP_DIR"
```

### Recuperação

```bash
# Recuperar modelos
cd /opt/Luna
tar -xzf /backup/luna/20240130/models.tar.gz

# Recuperar configuração
tar -xzf /backup/luna/20240130/config.tar.gz

# Verificar integridade
python verify_integration.py
python system_status.py
```

---

## Segurança

### Boas Práticas

1. **Não commitar secrets**
   ```bash
   # Adicionar ao .gitignore
   echo ".env" >> .gitignore
   echo "*.key" >> .gitignore
   echo "config/production.json" >> .gitignore
   ```

2. **Usar variáveis de ambiente**
   ```bash
   export WANDB_API_KEY=seu_key_aqui
   export OPENAI_API_KEY=seu_key_aqui
   ```

3. **Limitar acesso à API**
   ```python
   # Em src/web/app.py
   from flask_limiter import Limiter
   
   limiter = Limiter(
       app,
       key_func=lambda: request.remote_addr,
       default_limits=["200 per day", "50 per hour"]
   )
   ```

4. **HTTPS em produção**
   ```bash
   # Usar nginx como reverse proxy
   sudo apt install nginx
   # Configurar SSL com Let's Encrypt
   ```

---

## Checklist de Produção

Antes de colocar em produção, verificar:

- [ ] Sistema testado completamente
- [ ] Backups configurados
- [ ] Monitoramento ativo
- [ ] Logs configurados
- [ ] Segurança implementada
- [ ] Documentação atualizada
- [ ] Procedimentos de emergência definidos
- [ ] Equipe treinada
- [ ] Plano de rollback pronto
- [ ] Performance testada sob carga

---

## Contato e Suporte

- **Issues**: [GitHub Issues](https://github.com/SyraDevOps/Luna/issues)
- **Documentação**: Ver arquivos no repositório
- **Contribuições**: Ver CONTRIBUTING.md

---

**Luna GPT - Sistema Elite de Conversação AI**

*Sistema 1000/1000 como proposto* ✨
