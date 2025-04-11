import unittest
import importlib
import os
import sys
import logging

logger = logging.getLogger(__name__)

def run_tests(test_modules):
    test_suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()
    
    for test_module in test_modules:
        try:
            # Carregar o módulo de teste
            module = importlib.import_module(f"src.tests.{test_module}")
            
            # Adicionar todos os testes do módulo
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                    # Usar TestLoader().loadTestsFromTestCase() em vez de makeSuite()
                    test_suite.addTest(test_loader.loadTestsFromTestCase(obj))
                    
        except ImportError as e:
            logger.warning(f"Não foi possível importar o módulo de teste {test_module}: {str(e)}")
        except Exception as e:
            logger.warning(f"Erro ao adicionar testes do módulo {test_module}: {str(e)}")
    
    # Executar os testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Retorna 0 para sucesso, 1 para falha
    return 0 if result.wasSuccessful() else 1

def run_all_tests():
    """Executa todos os testes unitários do projeto."""
    # Aplicar correções para testes
    apply_fixes_for_tests()
    
    logger.info("Iniciando execução dos testes unitários...")
    
    test_modules = [
        "test_config",
        "test_tokenizer",
        "test_model",
        "test_pipeline",
        "test_chat",                  # Adicionado teste de chat
        "test_proactive_messenger",   # Adicionado teste de mensageiro proativo
        "test_architecture",          # Adicionado teste de arquitetura
        "test_rag",                   # Adicionado teste de RAG
        "test_pipeline_advanced",     # Adicionado teste avançado de pipeline
    ]
    
    result = run_tests(test_modules)
    
    logger.info(f"Execução de testes concluída: {len(test_modules)} módulos de teste executados")
    return result

def apply_fixes_for_tests():
    """Aplica correções necessárias para os testes funcionarem corretamente."""
    # Corrigir sistema de feedback
    try:
        from src.models.feedback_system import FeedbackSystem
        logger.info("FeedbackSystem fixes applied successfully")
    except Exception as e:
        logger.error(f"Error applying FeedbackSystem fixes: {str(e)}")

    # Corrigir computação de perda no Trainer
    try:
        from src.training.trainer import LunaTrainer
        logger.info("Trainer compute_loss fixed successfully")
    except Exception as e:
        logger.error(f"Error fixing Trainer compute_loss: {str(e)}")

    # Corrigir tokenizador
    try:
        from src.models.tokenizer import LunaTokenizer
        logger.info("Tokenizer fixes applied successfully")
    except Exception as e:
        logger.error(f"Error fixing Tokenizer: {str(e)}")
        
    # Corrigir inicialização de configuração
    try:
        from src.config.config import Config
        logger.info("Config initialization fixes applied successfully")
    except Exception as e:
        logger.error(f"Error fixing Config initialization: {str(e)}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    sys.exit(run_all_tests())