import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class HyperNetwork(nn.Module):
    """
    Gera dinamicamente parâmetros para outras camadas condicionados ao contexto,
    permitindo adaptação rápida a diferentes domínios ou tipos de entrada.
    """
    def __init__(self, context_dim, target_dim, hidden_dim=None):
        super().__init__()
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim or (context_dim * 2)
        
        # Rede que gera parâmetros de uma camada linear (pesos + bias)
        # para uma camada de tamanho target_dim
        self.weight_generator = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, target_dim)
        )
        
        # Gerador de bias (opcional)
        self.bias_generator = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
    def forward(self, context_vector):
        """
        Args:
            context_vector: Tensor de contexto [batch_size, context_dim]
            
        Returns:
            weight: Parâmetros gerados
            bias: Bias gerado
        """
        # Gerar os pesos dinamicamente com base no vetor de contexto
        weight = self.weight_generator(context_vector)
        bias = self.bias_generator(context_vector)
        
        return weight, bias
        
class HyperLinear(nn.Module):
    """
    Camada linear que usa parâmetros dinâmicos gerados por uma HyperNetwork
    """
    def __init__(self, input_dim, output_dim, context_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # HyperNetwork que gera os parâmetros
        # O alvo é uma matriz de weights (input_dim * output_dim)
        self.hypernet = HyperNetwork(
            context_dim=context_dim,
            target_dim=input_dim * output_dim
        )
        
        # Parâmetros estáticos como fallback ou regularização
        self.static_weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.02)
        self.static_bias = nn.Parameter(torch.zeros(output_dim))
        
        # Fator de mistura entre parâmetros estáticos e dinâmicos
        self.mix_factor = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, context_vector):
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            context_vector: Tensor de contexto [batch_size, context_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Gerar parâmetros dinâmicos
        dynamic_weight_flat, dynamic_bias = self.hypernet(context_vector)
        
        # Reformatar para dimensões corretas
        dynamic_weight = dynamic_weight_flat.view(batch_size, self.output_dim, self.input_dim)
        
        # Mesclar parâmetros estáticos e dinâmicos
        mix = torch.sigmoid(self.mix_factor)
        
        # Output para cada item no batch usando seus próprios parâmetros
        outputs = []
        for i in range(batch_size):
            # Combinar pesos estáticos e dinâmicos
            effective_weight = mix * self.static_weight + (1 - mix) * dynamic_weight[i]
            effective_bias = mix * self.static_bias + (1 - mix) * dynamic_bias[i]
            
            # Aplicar a transformação linear
            batch_output = torch.matmul(x[i], effective_weight.t()) + effective_bias
            outputs.append(batch_output)
            
        return torch.stack(outputs)