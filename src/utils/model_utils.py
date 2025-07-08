import logging

logger = logging.getLogger(__name__)

def expand_model_parameters(model, config, step):
    """Expande os parâmetros do modelo progressivamente."""
    try:
        if step % 1000 == 0:  # Expansão a cada 1000 passos
            new_hidden_size = min(config.hidden_size + 64, 1024)
            new_num_heads = min(config.num_attention_heads + 1, 16)
            if new_hidden_size > config.hidden_size or new_num_heads > config.num_attention_heads:
                logger.info(f"Expandindo parâmetros: hidden_size={new_hidden_size}, num_heads={new_num_heads}")
                
                # Atualizar configuração
                config.hidden_size = new_hidden_size
                config.num_attention_heads = new_num_heads
                model.config.hidden_size = config.hidden_size
                model.config.num_attention_heads = config.num_attention_heads
                
                # Inicializar novos pesos
                model.model.resize_token_embeddings(len(model.tokenizer))
                model.model.init_weights()
    except Exception as e:
        logger.error(f"Erro ao expandir parâmetros do modelo: {str(e)}")