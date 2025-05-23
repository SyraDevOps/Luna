import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
import math

logger = logging.getLogger(__name__)

class HyperNetwork(nn.Module):
    """
    Rede que gera pesos para outras redes baseado no contexto
    
    Uma hypernetwork é uma rede neural que gera os pesos para outra rede neural.
    Isso permite adaptação dinâmica baseada no contexto de entrada.
    """
    
    def __init__(
        self,
        context_size: int,
        target_layer_sizes: List[Tuple[int, int]],
        hyper_hidden_size: int = 256,
        compression_factor: int = 4,
        use_bias: bool = True
    ):
        """
        Inicializa a hypernetwork
        
        Args:
            context_size: Tamanho do vetor de contexto
            target_layer_sizes: Lista de (input_size, output_size) das camadas alvo
            hyper_hidden_size: Tamanho da camada oculta da hypernetwork
            compression_factor: Fator de compressão para reduzir parâmetros
            use_bias: Se deve gerar bias além dos pesos
        """
        super().__init__()
        
        self.context_size = context_size
        self.target_layer_sizes = target_layer_sizes
        self.hyper_hidden_size = hyper_hidden_size
        self.compression_factor = compression_factor
        self.use_bias = use_bias
        
        # Rede de contexto para processamento inicial
        self.context_processor = nn.Sequential(
            nn.Linear(context_size, hyper_hidden_size),
            nn.ReLU(),
            nn.Linear(hyper_hidden_size, hyper_hidden_size),
            nn.ReLU()
        )
        
        # Geradores de peso para cada camada alvo
        self.weight_generators = nn.ModuleList()
        self.bias_generators = nn.ModuleList() if use_bias else None
        
        for input_size, output_size in target_layer_sizes:
            # Calcular tamanho comprimido dos pesos
            weight_params = input_size * output_size
            compressed_size = max(weight_params // compression_factor, 64)
            
            # Gerador de pesos
            weight_gen = nn.Sequential(
                nn.Linear(hyper_hidden_size, compressed_size),
                nn.ReLU(),
                nn.Linear(compressed_size, weight_params)
            )
            self.weight_generators.append(weight_gen)
            
            # Gerador de bias se necessário
            if use_bias:
                bias_gen = nn.Sequential(
                    nn.Linear(hyper_hidden_size, output_size // 2 if output_size > 2 else output_size),
                    nn.ReLU(),
                    nn.Linear(output_size // 2 if output_size > 2 else output_size, output_size)
                )
                self.bias_generators.append(bias_gen)
        
        # Normalização de saída
        self.output_norm = nn.LayerNorm(hyper_hidden_size)
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos da hypernetwork"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, context: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Gera pesos baseado no contexto
        
        Args:
            context: Vetor de contexto (batch_size, context_size)
            
        Returns:
            Lista de dicionários com 'weight' e opcionalmente 'bias' para cada camada
        """
        batch_size = context.size(0)
        
        # Processar contexto
        context_features = self.context_processor(context)
        context_features = self.output_norm(context_features)
        
        generated_params = []
        
        for i, (input_size, output_size) in enumerate(self.target_layer_sizes):
            # Gerar pesos
            weight_flat = self.weight_generators[i](context_features)
            weight = weight_flat.view(batch_size, output_size, input_size)
            
            layer_params = {'weight': weight}
            
            # Gerar bias se necessário
            if self.use_bias and self.bias_generators:
                bias = self.bias_generators[i](context_features)
                layer_params['bias'] = bias
            
            generated_params.append(layer_params)
        
        return generated_params
    
    def get_parameter_count(self) -> int:
        """Retorna número total de parâmetros"""
        return sum(p.numel() for p in self.parameters())

class HyperConditionalLayer(nn.Module):
    """
    Camada que usa pesos gerados por hypernetwork
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        context_size: int,
        hyper_hidden_size: int = 256
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Hypernetwork para esta camada
        self.hypernet = HyperNetwork(
            context_size=context_size,
            target_layer_sizes=[(input_size, output_size)],
            hyper_hidden_size=hyper_hidden_size
        )
        
        # Camada base (fallback)
        self.base_layer = nn.Linear(input_size, output_size)
        
        # Controle de mistura
        self.mixing_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass com pesos condicionais
        
        Args:
            x: Entrada (batch_size, seq_len, input_size)
            context: Contexto (batch_size, context_size)
            
        Returns:
            Saída processada
        """
        batch_size, seq_len, _ = x.shape
        
        # Saída da camada base
        base_output = self.base_layer(x)
        
        # Gerar pesos condicionais
        hyper_params = self.hypernet(context)[0]
        hyper_weight = hyper_params['weight']  # (batch_size, output_size, input_size)
        hyper_bias = hyper_params.get('bias')   # (batch_size, output_size)
        
        # Aplicar pesos gerados
        x_flat = x.view(batch_size, seq_len, self.input_size)
        hyper_output = torch.bmm(x_flat, hyper_weight.transpose(1, 2))
        
        if hyper_bias is not None:
            hyper_output = hyper_output + hyper_bias.unsqueeze(1)
        
        # Misturar saídas
        mixing_weight = torch.sigmoid(self.mixing_weight)
        output = mixing_weight * hyper_output + (1 - mixing_weight) * base_output
        
        return output

class ContextualHyperNetwork(nn.Module):
    """
    Hypernetwork que adapta baseado no contexto da conversa
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        context_window: int = 10,
        num_context_types: int = 5
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context_window = context_window
        self.num_context_types = num_context_types
        
        # Analisador de contexto
        self.context_analyzer = nn.Sequential(
            nn.Linear(hidden_size * context_window, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_context_types),
            nn.Softmax(dim=-1)
        )
        
        # Embeddings de tipo de contexto
        self.context_embeddings = nn.Embedding(num_context_types, hidden_size)
        
        # Hypernetwork principal
        self.main_hypernet = HyperNetwork(
            context_size=hidden_size,
            target_layer_sizes=[(hidden_size, hidden_size), (hidden_size, vocab_size)],
            hyper_hidden_size=hidden_size * 2
        )
        
        # Cache de contexto
        self.register_buffer("context_cache", torch.zeros(1, context_window, hidden_size))
        self.cache_position = 0
    
    def update_context_cache(self, new_context: torch.Tensor):
        """Atualiza cache de contexto"""
        batch_size, seq_len, hidden_size = new_context.shape
        
        # Expandir cache se necessário
        if self.context_cache.size(0) < batch_size:
            new_cache = torch.zeros(batch_size, self.context_window, hidden_size, 
                                  device=self.context_cache.device)
            new_cache[:self.context_cache.size(0)] = self.context_cache
            self.context_cache = new_cache
        
        # Adicionar novo contexto
        for i in range(seq_len):
            self.context_cache[:batch_size, self.cache_position] = new_context[:, i]
            self.cache_position = (self.cache_position + 1) % self.context_window
    
    def get_current_context(self, batch_size: int) -> torch.Tensor:
        """Obtém contexto atual"""
        if batch_size > self.context_cache.size(0):
            # Expandir cache se necessário
            new_cache = torch.zeros(batch_size, self.context_window, self.hidden_size,
                                  device=self.context_cache.device)
            new_cache[:self.context_cache.size(0)] = self.context_cache
            self.context_cache = new_cache
        
        return self.context_cache[:batch_size]
    
    def forward(self, hidden_states: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass da hypernetwork contextual
        
        Args:
            hidden_states: Estados ocultos atuais
            
        Returns:
            Parâmetros gerados condicionalmente
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Atualizar cache de contexto
        self.update_context_cache(hidden_states)
        
        # Obter contexto atual
        current_context = self.get_current_context(batch_size)
        context_flat = current_context.view(batch_size, -1)
        
        # Analisar tipo de contexto
        context_type_probs = self.context_analyzer(context_flat)
        context_type_weighted = torch.matmul(context_type_probs, self.context_embeddings.weight)
        
        # Gerar parâmetros usando hypernetwork
        generated_params = self.main_hypernet(context_type_weighted)
        
        return generated_params

class AdaptiveAttentionHead(nn.Module):
    """
    Cabeça de atenção que adapta seus pesos baseado no contexto
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        head_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.context_size = context_size
        
        # Hypernetwork para projeções Q, K, V
        self.qkv_hypernet = HyperNetwork(
            context_size=context_size,
            target_layer_sizes=[
                (hidden_size, self.head_dim * num_heads),  # Q
                (hidden_size, self.head_dim * num_heads),  # K
                (hidden_size, self.head_dim * num_heads),  # V
            ],
            hyper_hidden_size=hidden_size
        )
        
        # Projeções base
        self.q_proj = nn.Linear(hidden_size, self.head_dim * num_heads)
        self.k_proj = nn.Linear(hidden_size, self.head_dim * num_heads)
        self.v_proj = nn.Linear(hidden_size, self.head_dim * num_heads)
        self.out_proj = nn.Linear(self.head_dim * num_heads, hidden_size)
        
        # Normalização
        self.scale = math.sqrt(self.head_dim)
        
        # Controle de adaptação
        self.adaptation_strength = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass da atenção adaptativa
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Projeções base
        q_base = self.q_proj(x)
        k_base = self.k_proj(x)
        v_base = self.v_proj(x)
        
        # Gerar adaptações baseadas no contexto
        if context is not None:
            hyper_params = self.qkv_hypernet(context)
            
            # Aplicar adaptações
            q_adapt = torch.bmm(x, hyper_params[0]['weight'].transpose(1, 2))
            k_adapt = torch.bmm(x, hyper_params[1]['weight'].transpose(1, 2))
            v_adapt = torch.bmm(x, hyper_params[2]['weight'].transpose(1, 2))
            
            # Misturar com projeções base
            strength = torch.sigmoid(self.adaptation_strength)
            q = q_base + strength * q_adapt
            k = k_base + strength * k_adapt
            v = v_base + strength * v_adapt
        else:
            q, k, v = q_base, k_base, v_base
        
        # Reshape para multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Atenção
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape de volta
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.head_dim * self.num_heads
        )
        
        # Projeção final
        output = self.out_proj(attn_output)
        
        return output