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

# Cache de modelos e chats carregados
model_cache = {}
chat_cache = {}

def create_app(model_name=None, config=None):
    """
    Cria e configura a aplicação Flask
    
    Args:
        model_name: Nome do modelo a ser usado por padrão
        config: Objeto de configuração Luna
        
    Returns:
        Flask app configurado
    """
    app = Flask(__name__, 
                static_folder="static",
                template_folder="templates")
    
    # Armazenar configurações na app
    app.config['DEFAULT_MODEL'] = model_name
    app.config['LUNA_CONFIG'] = config
    
    # Verificar importações de forma segura
    def safe_import():
        try:
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
    app.config['IMPORTS_OK'] = imports_ok
    
    def get_model(model_name_param):
        """Carrega ou retorna do cache o modelo solicitado"""
        if not imports_ok:
            logger.error("Módulos necessários não disponíveis")
            return None
            
        if model_name_param in model_cache:
            return model_cache[model_name_param]
        
        try:
            model_path = os.path.join("models", model_name_param)
            if not os.path.exists(model_path):
                logger.error(f"Modelo {model_name_param} não encontrado")
                return None
            
            logger.info(f"Carregando modelo {model_name_param}")
            
            # Usar config passado ou criar novo
            cfg = config if config else Config()
            model = LunaModel.from_pretrained(model_path, cfg)
            
            model_cache[model_name_param] = model
            return model
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_name_param}: {e}")
            return None
    
    def get_chat_instance(model_name_param):
        """Obtém ou cria uma instância de LunaChat para um modelo"""
        if model_name_param in chat_cache:
            return chat_cache[model_name_param]
        
        try:
            # Obter o modelo base
            model = get_model(model_name_param)
            if not model:
                return None
            
            # Importar LunaChat
            from src.chat.luna_chat import LunaChat
            
            # Usar configuração fornecida ou criar nova
            cfg = config if config else Config()
            
            # Criar instância de chat com o modelo carregado
            chat_instance = LunaChat(model_name_param, cfg, persona="casual")
            chat_cache[model_name_param] = chat_instance
            logger.info(f"Instância de chat criada para o modelo {model_name_param}")
            return chat_instance
        except Exception as e:
            logger.error(f"Erro ao criar instância de chat para {model_name_param}: {e}")
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
        
        default_model = app.config.get('DEFAULT_MODEL')
        
        return render_template('chat.html', 
                             models=available_models, 
                             now=now,
                             default_model=default_model,
                             imports_ok=imports_ok)
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        """Endpoint da API para processar mensagens de chat"""
        if not imports_ok:
            return jsonify({'error': 'Sistema não inicializado corretamente'}), 500
            
        data = request.json
        message = data.get('message', '')
        model_name_param = data.get('model', app.config.get('DEFAULT_MODEL', 'default'))
        
        if not message:
            return jsonify({'error': 'Mensagem vazia'}), 400
        
        # Obter instância de chat
        chat_instance = get_chat_instance(model_name_param)
        if not chat_instance:
            return jsonify({'error': f'Modelo {model_name_param} não disponível'}), 404
        
        # Capturar o "pensamento" do modelo
        thinking = []
        
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
            import traceback
            traceback.print_exc()
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
    
    @app.route('/api/health')
    def health():
        """Endpoint para verificar saúde da API"""
        return jsonify({
            'status': 'ok',
            'imports_ok': imports_ok,
            'default_model': app.config.get('DEFAULT_MODEL'),
            'models_loaded': list(model_cache.keys()),
            'chats_active': list(chat_cache.keys())
        })
    
    return app


# Para execução standalone
if __name__ == '__main__':
    app = create_app()
    
    if app.config.get('IMPORTS_OK'):
        logger.info("Sistema inicializado com sucesso")
    else:
        logger.warning("Interface web iniciada em modo limitado - alguns recursos podem não funcionar")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
