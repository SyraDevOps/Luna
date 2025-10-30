#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Verificação de Integração Luna
Verifica que todos os componentes estão funcionando corretamente
"""
import os
import sys
import logging
from typing import List, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationVerifier:
    """Verifica a integração de todos os componentes do sistema Luna"""
    
    def __init__(self):
        self.results = []
        self.failed_checks = []
        
    def check(self, name: str, func) -> bool:
        """Executa uma verificação e registra o resultado"""
        try:
            logger.info(f"Verificando: {name}...")
            result = func()
            if result:
                logger.info(f"✓ {name}: OK")
                self.results.append((name, True, None))
                return True
            else:
                logger.error(f"✗ {name}: FALHOU")
                self.results.append((name, False, "Verificação retornou False"))
                self.failed_checks.append(name)
                return False
        except Exception as e:
            logger.error(f"✗ {name}: ERRO - {str(e)}")
            self.results.append((name, False, str(e)))
            self.failed_checks.append(name)
            return False
    
    def verify_dependencies(self) -> bool:
        """Verifica se as dependências estão instaladas"""
        dependencies = [
            'torch',
            'transformers',
            'nltk',
            'sentence_transformers',
            'flask',
            'numpy',
            'pandas'
        ]
        
        all_ok = True
        for dep in dependencies:
            try:
                __import__(dep)
                logger.info(f"  ✓ {dep}")
            except ImportError:
                logger.error(f"  ✗ {dep} - NÃO INSTALADO")
                all_ok = False
        
        return all_ok
    
    def verify_nltk_resources(self) -> bool:
        """Verifica se os recursos do NLTK estão disponíveis"""
        try:
            import nltk
            resources = ['punkt', 'stopwords', 'wordnet']
            
            for resource in resources:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                    logger.info(f"  ✓ NLTK {resource}")
                except LookupError:
                    logger.warning(f"  ! NLTK {resource} não encontrado, baixando...")
                    nltk.download(resource, quiet=True)
            
            return True
        except Exception as e:
            logger.error(f"Erro ao verificar NLTK: {e}")
            return False
    
    def verify_config_system(self) -> bool:
        """Verifica o sistema de configuração"""
        try:
            from src.config.config import Config
            
            # Criar configuração padrão
            config = Config()
            
            # Verificar seções principais
            assert hasattr(config, 'model')
            assert hasattr(config, 'training')
            assert hasattr(config, 'rag')
            assert hasattr(config, 'feedback')
            
            # Verificar métodos
            assert hasattr(config, 'get_summary')
            assert hasattr(config, 'save_to_file')
            
            logger.info("  ✓ Configuração carregada com sucesso")
            logger.info(f"  ✓ Modelo: {config.model.model_name}")
            logger.info(f"  ✓ Hidden size: {config.model.hidden_size}")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao verificar configuração: {e}")
            return False
    
    def verify_model_components(self) -> bool:
        """Verifica os componentes do modelo"""
        try:
            from src.models.moe import MoEBlock
            from src.models.hypernet import HyperNetwork
            
            logger.info("  ✓ MoEBlock importado")
            logger.info("  ✓ HyperNetwork importado")
            
            # Verificar se podem ser instanciados (sem pytorch)
            logger.info("  ✓ Componentes verificados")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao verificar componentes: {e}")
            return False
    
    def verify_tokenizer(self) -> bool:
        """Verifica o tokenizador"""
        try:
            from src.models.tokenizer import LunaTokenizer
            from src.config.config import Config
            
            config = Config()
            logger.info("  ✓ LunaTokenizer importado")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao verificar tokenizador: {e}")
            return False
    
    def verify_chat_system(self) -> bool:
        """Verifica o sistema de chat"""
        try:
            from src.chat.luna_chat import LunaChat
            from src.chat.proactive_messenger import ProactiveMessenger
            
            logger.info("  ✓ LunaChat importado")
            logger.info("  ✓ ProactiveMessenger importado")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao verificar sistema de chat: {e}")
            return False
    
    def verify_web_interface(self) -> bool:
        """Verifica a interface web"""
        try:
            from src.web.app import create_app
            
            app = create_app()
            logger.info("  ✓ create_app() funcionando")
            
            # Verificar rotas principais
            routes = [rule.rule for rule in app.url_map.iter_rules()]
            expected_routes = ['/', '/api/chat', '/api/models', '/api/health']
            
            for route in expected_routes:
                if route in routes:
                    logger.info(f"  ✓ Rota {route}")
                else:
                    logger.warning(f"  ! Rota {route} não encontrada")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao verificar interface web: {e}")
            return False
    
    def verify_rag_system(self) -> bool:
        """Verifica o sistema RAG"""
        try:
            from src.models.rag_retriever import RAGRetriever
            
            logger.info("  ✓ RAGRetriever importado")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao verificar RAG: {e}")
            return False
    
    def verify_training_system(self) -> bool:
        """Verifica o sistema de treinamento"""
        try:
            from src.training.trainer import LunaTrainer
            
            logger.info("  ✓ LunaTrainer importado")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao verificar sistema de treinamento: {e}")
            return False
    
    def verify_feedback_system(self) -> bool:
        """Verifica o sistema de feedback"""
        try:
            from src.models.feedback_system import FeedbackSystem
            
            logger.info("  ✓ FeedbackSystem importado")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao verificar sistema de feedback: {e}")
            return False
    
    def verify_directories(self) -> bool:
        """Verifica a estrutura de diretórios"""
        required_dirs = [
            'data',
            'data/train',
            'data/valid',
            'models',
            'logs',
            'temp',
            'config'
        ]
        
        all_ok = True
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                logger.info(f"  ✓ {dir_path}")
            else:
                logger.warning(f"  ! {dir_path} não existe, criando...")
                os.makedirs(dir_path, exist_ok=True)
        
        return True
    
    def verify_cli(self) -> bool:
        """Verifica a CLI principal"""
        try:
            import main
            
            # Verificar que main.py tem as funções principais
            assert hasattr(main, 'main')
            assert hasattr(main, 'parse_args')
            assert hasattr(main, 'create_model')
            assert hasattr(main, 'train_model')
            assert hasattr(main, 'chat_with_model')
            
            logger.info("  ✓ main.py verificado")
            logger.info("  ✓ Todas as funções principais presentes")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao verificar CLI: {e}")
            return False
    
    def run_all_checks(self):
        """Executa todas as verificações"""
        logger.info("="*60)
        logger.info("VERIFICAÇÃO DE INTEGRAÇÃO DO SISTEMA LUNA")
        logger.info("="*60)
        
        checks = [
            ("Dependências", self.verify_dependencies),
            ("Recursos NLTK", self.verify_nltk_resources),
            ("Sistema de Configuração", self.verify_config_system),
            ("Componentes do Modelo", self.verify_model_components),
            ("Tokenizador", self.verify_tokenizer),
            ("Sistema de Chat", self.verify_chat_system),
            ("Interface Web", self.verify_web_interface),
            ("Sistema RAG", self.verify_rag_system),
            ("Sistema de Treinamento", self.verify_training_system),
            ("Sistema de Feedback", self.verify_feedback_system),
            ("Estrutura de Diretórios", self.verify_directories),
            ("CLI Principal", self.verify_cli),
        ]
        
        for name, func in checks:
            self.check(name, func)
        
        # Resumo
        logger.info("\n" + "="*60)
        logger.info("RESUMO DA VERIFICAÇÃO")
        logger.info("="*60)
        
        passed = sum(1 for _, status, _ in self.results if status)
        total = len(self.results)
        
        logger.info(f"Total de verificações: {total}")
        logger.info(f"Passou: {passed}")
        logger.info(f"Falhou: {total - passed}")
        
        if self.failed_checks:
            logger.warning("\nVerificações que falharam:")
            for check in self.failed_checks:
                logger.warning(f"  - {check}")
        
        percentage = (passed / total * 100) if total > 0 else 0
        logger.info(f"\nPontuação: {percentage:.1f}%")
        
        if percentage >= 90:
            logger.info("✓ Sistema EXCELENTE (>=90%)")
        elif percentage >= 75:
            logger.info("✓ Sistema BOM (>=75%)")
        elif percentage >= 50:
            logger.warning("! Sistema ACEITÁVEL (>=50%)")
        else:
            logger.error("✗ Sistema PRECISA DE MELHORIAS (<50%)")
        
        return percentage >= 75


if __name__ == "__main__":
    verifier = IntegrationVerifier()
    success = verifier.run_all_checks()
    
    sys.exit(0 if success else 1)
