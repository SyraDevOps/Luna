import torch
import torch.nn as nn
import logging
import math
import numpy as np
from typing import Dict, List, Union, Optional, Tuple

logger = logging.getLogger(__name__)

class GrowingNetwork(nn.Module):
    """
    Expande a arquitetura durante o treinamento, adicionando camadas
    conforme o modelo atinge certos marcos de performance.
    """
    def __init__(self, base_model: nn.Module, growth_config: Dict = None):
        super().__init__()
        self.base_model = base_model
        self.additional_layers = nn.ModuleList()
        
        # Configuração para crescimento
        self.growth_config = growth_config or {
            'growth_trigger': 'loss',  # 'loss', 'accuracy', 'epoch'
            'growth_threshold': 0.01,  # Melhoria mínima para adicionar camadas
            'max_additional_layers': 3,
            'growth_dimensions': [256, 384, 512],  # Dimensões para novas camadas
            'activation': 'gelu'  # gelu, relu, swish
        }
        
        # Histórico de métricas para determinar crescimento
        self.metric_history = []
        self.growth_count = 0
        
    def add_layer(self, input_dim: int, output_dim: int):
        """
        Adiciona uma nova camada à rede
        """
        if self.growth_count >= self.growth_config['max_additional_layers']:
            logger.info("Limite máximo de camadas adicionais atingido.")
            return False
            
        # Criar nova camada
        activation_fn = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU()  # Swish/SiLU activation
        }.get(self.growth_config['activation'], nn.GELU())
        
        new_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            activation_fn,
            nn.Dropout(0.1)
        )
        
        # Adicionar à lista de camadas
        self.additional_layers.append(new_layer)
        self.growth_count += 1
        logger.info(f"Nova camada adicionada: {input_dim} -> {output_dim} (Total: {self.growth_count})")
        return True
        
    def should_grow(self, current_metric: float) -> bool:
        """
        Determina se a rede deve crescer baseado no histórico de métricas
        """
        if not self.metric_history:
            self.metric_history.append(current_metric)
            return False
            
        # Verificar se houve melhoria suficiente para adicionar camada
        last_best = min(self.metric_history) if self.growth_config['growth_trigger'] == 'loss' else max(self.metric_history)
        improvement = (last_best - current_metric) if self.growth_config['growth_trigger'] == 'loss' else (current_metric - last_best)
        
        # Adicionar ao histórico
        self.metric_history.append(current_metric)
        
        # Verificar se melhoria atinge o limiar
        if improvement > self.growth_config['growth_threshold']:
            return self.growth_count < self.growth_config['max_additional_layers']
            
        return False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass incluindo camadas adicionais
        """
        x = self.base_model(x)
        
        # Aplicar camadas adicionais em sequência
        for layer in self.additional_layers:
            x = layer(x)
            
        return x


class StateSpaceLayer(nn.Module):
    """Camada StateSpace simples para adaptação dinâmica"""
    
    def __init__(self, input_dim=None, hidden_size=None, state_size=None):
        """
        Inicializa uma camada StateSpace.
        
        Args:
            input_dim: (compatibilidade) Dimensão de entrada
            hidden_size: Dimensão oculta da camada
            state_size: Dimensão do estado interno
        """
        super().__init__()
        # Usar input_dim como fallback para hidden_size
        self.hidden_size = hidden_size if hidden_size is not None else input_dim
        self.state_size = state_size if state_size is not None else (self.hidden_size // 4)
        
        # State-space parameters
        self.A = nn.Parameter(torch.randn(self.state_size, self.state_size) * 0.01)
        # Corrigir dimensões das matrizes B e C para multiplicação compatível
        self.B = nn.Parameter(torch.randn(self.state_size, self.state_size) * 0.01)  # state_size x state_size
        self.C = nn.Parameter(torch.randn(self.hidden_size, self.state_size) * 0.01)
        self.D = nn.Parameter(torch.zeros(self.hidden_size) + 0.01)
        
        # Linear projections
        self.in_proj = nn.Linear(self.hidden_size, self.state_size)
        self.out_proj = nn.Linear(self.state_size, self.hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Project input
        u = self.in_proj(x)  # [batch, seq_len, state_size]
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.state_size, device=x.device)
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            ut = u[:, t, :]  # [batch, state_size]
            
            # State update: h_t = A*h_{t-1} + B*u_t
            # Crucial: transposição correta para compatibilidade dimensional
            A_expanded = self.A.expand(batch_size, self.state_size, self.state_size)
            h = torch.bmm(h.unsqueeze(1), A_expanded).squeeze(1) + torch.matmul(ut, self.B.t())
            
            # Output: y_t = C*h_t + D*u_t
            y = torch.matmul(h, self.C.t()) + self.D
            outputs.append(y)
        
        # Stack outputs
        out = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden_size]
        
        return out