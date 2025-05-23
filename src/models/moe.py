import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class Expert(nn.Module):
    """
    Especialista individual para o sistema MoE
    """
    def __init__(self, hidden_size: int, expert_hidden_size: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        
        self.up_proj = nn.Linear(hidden_size, expert_hidden_size)
        self.down_proj = nn.Linear(expert_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Função de ativação
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "swish":
            self.activation = F.silu
        else:
            self.activation = F.gelu
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa os pesos do especialista"""
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.down_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass do especialista"""
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x

class MoEBlock(nn.Module):
    """
    Mixture of Experts (MoE) Block para o LunaGPT
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int = 8, 
        top_k: int = 2, 
        expert_hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_hidden_size = expert_hidden_size or 4 * hidden_size
        self.dropout = dropout
        self.load_balancing_weight = load_balancing_weight
        
        # Gateway/Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Especialistas
        self.experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                expert_hidden_size=self.expert_hidden_size,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_experts)
        ])
        
        # Normalização
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Estatísticas de uso
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer("total_tokens", torch.tensor(0.0))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa os pesos do módulo"""
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass do MoE"""
        batch_size, seq_len, hidden_size = x.shape
        original_shape = x.shape
        
        # Flatten para processamento
        x_flat = x.view(-1, hidden_size)
        
        # Normalizar entrada
        x_norm = self.layer_norm(x_flat)
        
        # Calcular scores do gateway
        gate_scores = self.gate(x_norm)
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # Selecionar top-k especialistas
        top_k_scores, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Normalizar scores
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Preparar saída
        output = torch.zeros_like(x_flat)
        
        # Processar cada token através dos especialistas selecionados
        for i in range(x_flat.size(0)):
            token_output = torch.zeros_like(x_flat[i])
            
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j].item()
                expert_weight = top_k_scores[i, j]
                
                expert_output = self.experts[expert_idx](x_norm[i].unsqueeze(0))
                token_output += expert_weight * expert_output.squeeze(0)
                
                # Atualizar estatísticas
                self.expert_usage[expert_idx] += 1
            
            output[i] = token_output
        
        self.total_tokens += x_flat.size(0)
        
        # Reshape de volta
        output = output.view(original_shape)
        
        # Calcular perda de balanceamento
        load_balancing_loss = self._compute_load_balancing_loss(gate_probs, top_k_indices)
        
        # Informações auxiliares
        aux_info = {
            "gate_scores": gate_scores.view(batch_size, seq_len, self.num_experts),
            "top_k_indices": top_k_indices.view(batch_size, seq_len, self.top_k),
            "top_k_scores": top_k_scores.view(batch_size, seq_len, self.top_k),
            "load_balancing_loss": load_balancing_loss,
            "expert_usage": self.expert_usage.clone()
        }
        
        return output, aux_info
    
    def _compute_load_balancing_loss(self, gate_probs: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """Computa perda de balanceamento de carga"""
        # Frequência de uso de cada especialista
        expert_counts = torch.zeros(self.num_experts, device=gate_probs.device)
        
        for i in range(self.num_experts):
            expert_counts[i] = (top_k_indices == i).float().sum()
        
        # Normalizar
        expert_freq = expert_counts / top_k_indices.numel()
        
        # Probabilidade média de cada especialista
        expert_probs = gate_probs.mean(dim=0)
        
        # Perda de balanceamento (CV loss)
        loss = self.num_experts * torch.sum(expert_freq * expert_probs)
        
        return self.load_balancing_weight * loss
    
    def get_expert_usage_stats(self) -> Dict[str, float]:
        """Retorna estatísticas de uso dos especialistas"""
        if self.total_tokens > 0:
            usage_percentages = (self.expert_usage / self.total_tokens * 100).cpu().numpy()
            return {
                f"expert_{i}": float(usage_percentages[i])
                for i in range(self.num_experts)
            }
        return {}
    
    def reset_usage_stats(self):
        """Reseta estatísticas de uso"""
        self.expert_usage.zero_()
        self.total_tokens.zero_()

class SparseMoELayer(nn.Module):
    """
    Camada MoE esparsa integrada com atenção
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.moe_block = MoEBlock(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_size=expert_hidden_size,
            dropout=dropout
        )
        
        # Conexão residual
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass com conexão residual"""
        moe_output, aux_info = self.moe_block(x)
        
        # Conexão residual
        output = x + self.dropout(moe_output)
        
        return output, aux_info

def create_moe_config(
    hidden_size: int,
    num_layers: int,
    moe_layers: Optional[list] = None,
    num_experts: int = 8,
    top_k: int = 2
) -> Dict[str, Any]:
    """
    Cria configuração para modelo com MoE
    
    Args:
        hidden_size: Tamanho da dimensão oculta
        num_layers: Número total de camadas
        moe_layers: Lista de índices das camadas MoE (se None, usa camadas alternadas)
        num_experts: Número de especialistas
        top_k: Número de especialistas ativos
    
    Returns:
        Configuração do MoE
    """
    if moe_layers is None:
        # Usar camadas alternadas (cada segunda camada é MoE)
        moe_layers = list(range(1, num_layers, 2))
    
    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "moe_layers": moe_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_hidden_size": 4 * hidden_size,
        "load_balancing_weight": 0.01
    }