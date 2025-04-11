import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MoEBlock(nn.Module):
    """Bloco de Mixture of Experts para melhor especialização"""
    
    def __init__(self, input_dim, num_experts=4, hidden_dim=None, 
                sparse_top_k=2, emotional_routing=False, emotional_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim if hidden_dim else input_dim * 4
        self.sparse_top_k = sparse_top_k
        self.emotional_routing = emotional_routing
        self.emotional_dim = emotional_dim
        
        # Criar experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Router network
        self.router = nn.Linear(input_dim, num_experts)
        
        # Emotional routing components
        if emotional_routing:
            self.emotional_router = nn.Linear(emotional_dim, num_experts)
    
    def forward(self, x, emotion_weights=None):
        """
        Forward pass do MoE.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, input_dim]
            emotion_weights: Pesos emocionais opcionais [batch_size, emotional_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape para processamento por especialista
        x_flat = x.reshape(-1, self.input_dim)
        
        # Calcular pesos de roteamento
        routing_logits = self.router(x_flat)
        
        # Aplicar roteamento emocional se fornecido
        if self.emotional_routing and emotion_weights is not None:
            # Expandir emotion_weights para cada token na sequência
            emotion_logits = self.emotional_router(emotion_weights)  # [batch_size, num_experts]
            emotion_logits = emotion_logits.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, num_experts]
            emotion_logits = emotion_logits.reshape(-1, self.num_experts)  # [batch_size*seq_len, num_experts]
            
            # Combinar com os logits do router
            routing_logits = routing_logits + emotion_logits
        
        # Aplicar top-k sparsity
        if self.sparse_top_k < self.num_experts:
            # Zerar todos exceto os top-k experts
            routing_logits_sorted, _ = torch.sort(routing_logits, dim=-1, descending=True)
            routing_logits_threshold = routing_logits_sorted[:, self.sparse_top_k-1:self.sparse_top_k]
            routing_logits = torch.where(routing_logits >= routing_logits_threshold, routing_logits, 
                                        torch.ones_like(routing_logits) * float('-inf'))
        
        # Normalizar pesos para soma = 1
        routing_weights = torch.softmax(routing_logits, dim=-1)
        
        # Computar outputs de cada especialista
        expert_outputs = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            expert_output = expert(x_flat)
            expert_outputs += expert_output * routing_weights[:, i].unsqueeze(-1)
        
        # Reshape de volta para forma original
        output = expert_outputs.reshape(batch_size, seq_len, self.input_dim)
        return output