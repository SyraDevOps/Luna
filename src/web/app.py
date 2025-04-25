from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
from datetime import datetime
import importlib.util

# Adicionar o diretório raiz ao path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_folder="static",
            template_folder="templates")

# Verificar importações de forma segura
def safe_import():
    try:
        # Verificar se o módulo luna_model está disponível
        if importlib.util.find_spec("src.models.luna_model") is not None:
            from src.models.luna_model import LunaModel
            from src.config.config import Config
            return True, LunaModel, Config
        else:
            logger.error("Módulo luna_model não encontrado")
            return False, None, None
    except Exception as e:
        logger.error(f"Erro ao importar módulos: {e}")
        return False, None, None

# Carregar classes necessárias
imports_ok, LunaModel, Config = safe_import()

# Cache de modelos carregados
model_cache = {}

def get_model(model_name):
    """Carrega ou retorna do cache o modelo solicitado"""
    if not imports_ok:
        logger.error("Módulos necessários não disponíveis")
        return None
        
    if model_name in model_cache:
        return model_cache[model_name]
    
    try:
        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            logger.error(f"Modelo {model_name} não encontrado")
            return None
        
        logger.info(f"Carregando modelo {model_name}")
        # CORREÇÃO: Usar from_pretrained em vez do construtor
        model = LunaModel.from_pretrained(model_path)
        model_cache[model_name] = model
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo {model_name}: {e}")
        return None

# Na função que carrega o modelo, adicione um dicionário para armazenar instâncias de LunaChat
chat_cache = {}  # Cache para instâncias de LunaChat

def get_chat_instance(model_name):
    """Obtém ou cria uma instância de LunaChat para um modelo"""
    if model_name in chat_cache:
        return chat_cache[model_name]
    
    try:
        # Obter o modelo base
        model = get_model(model_name)
        if not model:
            return None
        
        # Importar LunaChat
        from src.chat.luna_chat import LunaChat
        from src.config.config import Config
        
        # Criar configuração básica
        config = Config()
        
        # Criar instância de chat com o modelo carregado
        chat_instance = LunaChat(model_name, config, persona="casual")
        chat_cache[model_name] = chat_instance
        logger.info(f"Instância de chat criada para o modelo {model_name}")
        return chat_instance
    except Exception as e:
        logger.error(f"Erro ao criar instância de chat para {model_name}: {e}")
        return None

@app.route('/')
def index():
    """Página principal do chat"""
    # Listar modelos disponíveis
    models_dir = os.path.join(os.getcwd(), "models")
    available_models = []
    
    if os.path.exists(models_dir):
        available_models = [d for d in os.listdir(models_dir) 
                            if os.path.isdir(os.path.join(models_dir, d))]
    
    # Adicionar data atual para o timestamp inicial
    now = datetime.now().strftime("%H:%M")
    
    return render_template('chat.html', models=available_models, now=now, 
                          imports_ok=imports_ok)  # Pass imports status to template

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint da API para processar mensagens de chat"""
    if not imports_ok:
        return jsonify({'error': 'Sistema não inicializado corretamente'}), 500
        
    data = request.json
    message = data.get('message', '')
    model_name = data.get('model', 'default')
    
    if not message:
        return jsonify({'error': 'Mensagem vazia'}), 400
    
    # Obter instância de chat em vez do modelo diretamente
    chat_instance = get_chat_instance(model_name)
    if not chat_instance:
        return jsonify({'error': f'Modelo {model_name} não disponível'}), 404
    
    # Capturar o "pensamento" do modelo
    thinking = []
    def thinking_callback(step, content):
        thinking.append(f"Passo {step}: {content}")
    
    # Gerar resposta
    try:
        start_time = datetime.now()
        
        # Adicionar lógica para capturar o raciocínio
        thinking.append("Analisando entrada do usuário...")
        thinking.append(f"Processando: '{message}'")
        
        # Chamar o método generate_response da instância de chat
        response = chat_instance.generate_response(message)
        
        end_time = datetime.now()
        process_time = (end_time - start_time).total_seconds()
        
        thinking.append(f"Resposta gerada em {process_time:.2f} segundos")
        
        return jsonify({
            'response': response,
            'thinking': '\n'.join(thinking),
            'processTime': process_time
        })
    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {e}")
        return jsonify({
            'error': 'Erro ao processar mensagem',
            'details': str(e)
        }), 500

@app.route('/api/models')
def list_models():
    """Lista os modelos disponíveis"""
    models_dir = os.path.join(os.getcwd(), "models")
    available_models = []
    
    if os.path.exists(models_dir):
        available_models = [d for d in os.listdir(models_dir) 
                           if os.path.isdir(os.path.join(models_dir, d))]
    
    return jsonify({'models': available_models})

if __name__ == '__main__':
    if imports_ok:
        logger.info("Sistema inicializado com sucesso")
    else:
        logger.warning("Interface web iniciada em modo limitado - alguns recursos podem não funcionar")
    
    app.run(debug=False, port=5000)  # Set debug=False to avoid double initialization