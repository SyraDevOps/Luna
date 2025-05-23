import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class GrowingNetwork(nn.Module):
    """
    Rede Neural que cresce dinamicamente baseada na demanda e performance
    
    Este módulo implementa um sistema que pode:
    - Adicionar novos neurônios quando necessário
    - Remover neurônios pouco utilizados
    - Ajustar a arquitetura baseada no feedback
    """
    
    def __init__(
        self,
        initial_size: int,
        max_size: int,
        growth_rate: float = 0.1,
        prune_threshold: float = 0.01,
        performance_threshold: float = 0.8,
        growth_interval: int = 1000
    ):
        """
        Inicializa a rede crescente
        
        Args:
            initial_size: Tamanho inicial da camada
            max_size: Tamanho máximo permitido
            growth_rate: Taxa de crescimento (percentual)
            prune_threshold: Threshold para remoção de neurônios
            performance_threshold: Threshold de performance para crescimento
            growth_interval: Intervalo de steps para avaliação de crescimento
        """
        super().__init__()
        
        self.initial_size = initial_size
        self.max_size = max_size
        self.growth_rate = growth_rate
        self.prune_threshold = prune_threshold
        self.performance_threshold = performance_threshold
        self.growth_interval = growth_interval
        
        # Camada principal
        self.linear = nn.Linear(initial_size, initial_size)
        self.current_size = initial_size
        
        # Estatísticas de neurônios
        self.register_buffer("neuron_activations", torch.zeros(initial_size))
        self.register_buffer("neuron_gradients", torch.zeros(initial_size))
        self.register_buffer("neuron_usage_count", torch.zeros(initial_size))
        self.register_buffer("step_count", torch.tensor(0))
        
        # Histórico de performance
        self.performance_history = []
        self.growth_history = []
        
        # Normalização
        self.layer_norm = nn.LayerNorm(initial_size)
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos da rede"""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, track_usage: bool = True) -> torch.Tensor:
        """
        Forward pass com rastreamento de uso
        
        Args:
            x: Tensor de entrada
            track_usage: Se deve rastrear uso dos neurônios
        
        Returns:
            Saída processada
        """
        # Normalizar entrada
        x_norm = self.layer_norm(x)
        
        # Forward pass
        output = self.linear(x_norm)
        
        # Rastrear uso se solicitado
        if track_usage and self.training:
            self._track_neuron_usage(output)
        
        self.step_count += 1
        
        # Verificar se é hora de crescer
        if self.step_count % self.growth_interval == 0 and self.training:
            self._evaluate_growth()
        
        return output
    
    def _track_neuron_usage(self, output: torch.Tensor):
        """Rastreia uso e ativação dos neurônios"""
        with torch.no_grad():
            # Ativações médias
            activations = output.abs().mean(dim=0)
            self.neuron_activations = 0.9 * self.neuron_activations + 0.1 * activations
            
            # Contador de uso (neurônios ativos)
            active_neurons = (output.abs() > 1e-6).float().mean(dim=0)
            self.neuron_usage_count += active_neurons
    
    def _track_gradients(self):
        """Rastreia gradientes dos neurônios"""
        if self.linear.weight.grad is not None:
            with torch.no_grad():
                grad_norms = self.linear.weight.grad.norm(dim=0)
                self.neuron_gradients = 0.9 * self.neuron_gradients + 0.1 * grad_norms
    
    def _evaluate_growth(self):
        """Avalia se a rede deve crescer ou ser podada"""
        try:
            # Avaliar performance recente
            avg_performance = self._get_recent_performance()
            
            # Decidir ação baseada na performance
            if avg_performance < self.performance_threshold and self.current_size < self.max_size:
                # Performance baixa - considerar crescimento
                self._grow_network()
            elif len(self.performance_history) > 10:
                # Performance boa - considerar poda
                self._prune_network()
                
        except Exception as e:
            logger.warning(f"Erro durante avaliação de crescimento: {str(e)}")
    
    def _get_recent_performance(self) -> float:
        """Calcula performance média recente"""
        if len(self.performance_history) < 5:
            return 0.5  # Performance neutra se histórico insuficiente
        
        recent_performance = self.performance_history[-5:]
        return sum(recent_performance) / len(recent_performance)
    
    def _grow_network(self):
        """Adiciona novos neurônios à rede"""
        if self.current_size >= self.max_size:
            return
        
        # Calcular quantos neurônios adicionar
        growth_amount = max(1, int(self.current_size * self.growth_rate))
        new_size = min(self.current_size + growth_amount, self.max_size)
        
        if new_size <= self.current_size:
            return
        
        logger.info(f"Crescendo rede de {self.current_size} para {new_size} neurônios")
        
        # Criar nova camada linear com tamanho maior
        old_linear = self.linear
        self.linear = nn.Linear(new_size, new_size).to(old_linear.weight.device)
        
        # Copiar pesos existentes
        with torch.no_grad():
            # Copiar pesos de entrada
            self.linear.weight[:self.current_size, :self.current_size] = old_linear.weight
            
            # Inicializar novos pesos
            new_weights = self.linear.weight[self.current_size:, :]
            nn.init.xavier_uniform_(new_weights)
            
            # Copiar bias
            self.linear.bias[:self.current_size] = old_linear.bias
            
            # Inicializar novos bias
            nn.init.zeros_(self.linear.bias[self.current_size:])
        
        # Expandir buffers de rastreamento
        old_size = self.current_size
        self.current_size = new_size
        
        # Expandir tensors de rastreamento
        new_activations = torch.zeros(new_size, device=self.neuron_activations.device)
        new_activations[:old_size] = self.neuron_activations
        self.neuron_activations = new_activations
        
        new_gradients = torch.zeros(new_size, device=self.neuron_gradients.device)
        new_gradients[:old_size] = self.neuron_gradients
        self.neuron_gradients = new_gradients
        
        new_usage = torch.zeros(new_size, device=self.neuron_usage_count.device)
        new_usage[:old_size] = self.neuron_usage_count
        self.neuron_usage_count = new_usage
        
        # Atualizar layer norm
        self.layer_norm = nn.LayerNorm(new_size).to(old_linear.weight.device)
        
        # Registrar crescimento
        self.growth_history.append({
            'step': self.step_count.item(),
            'action': 'grow',
            'old_size': old_size,
            'new_size': new_size
        })
    
    def _prune_network(self):
        """Remove neurônios pouco utilizados"""
        if self.current_size <= self.initial_size:
            return  # Não podar abaixo do tamanho inicial
        
        # Identificar neurônios para remoção
        usage_scores = self.neuron_activations + self.neuron_gradients
        avg_usage = usage_scores.mean()
        
        # Neurônios com uso muito baixo
        prune_mask = usage_scores < (avg_usage * self.prune_threshold)
        prune_indices = prune_mask.nonzero().squeeze(-1)
        
        if len(prune_indices) == 0:
            return
        
        # Não remover muitos neurônios de uma vez
        max_prune = max(1, int(self.current_size * 0.1))
        if len(prune_indices) > max_prune:
            # Manter os mais utilizados entre os candidatos à poda
            _, keep_indices = torch.topk(usage_scores[prune_indices], 
                                       len(prune_indices) - max_prune)
            prune_indices = prune_indices[torch.tensor([i for i in range(len(prune_indices)) 
                                                      if i not in keep_indices])]
        
        logger.info(f"Podando {len(prune_indices)} neurônios da rede")
        
        # Criar máscara de neurônios a manter
        keep_indices = torch.tensor([i for i in range(self.current_size) 
                                   if i not in prune_indices])
        
        new_size = len(keep_indices)
        
        # Criar nova camada linear
        old_linear = self.linear
        self.linear = nn.Linear(new_size, new_size).to(old_linear.weight.device)
        
        # Copiar pesos dos neurônios mantidos
        with torch.no_grad():
            self.linear.weight = nn.Parameter(old_linear.weight[keep_indices][:, keep_indices])
            self.linear.bias = nn.Parameter(old_linear.bias[keep_indices])
        
        # Atualizar buffers
        self.neuron_activations = self.neuron_activations[keep_indices]
        self.neuron_gradients = self.neuron_gradients[keep_indices]
        self.neuron_usage_count = self.neuron_usage_count[keep_indices]
        
        # Atualizar layer norm
        self.layer_norm = nn.LayerNorm(new_size).to(old_linear.weight.device)
        
        old_size = self.current_size
        self.current_size = new_size
        
        # Registrar poda
        self.growth_history.append({
            'step': self.step_count.item(),
            'action': 'prune',
            'old_size': old_size,
            'new_size': new_size,
            'pruned_neurons': len(prune_indices)
        })
    
    def update_performance(self, performance_score: float):
        """Atualiza histórico de performance"""
        self.performance_history.append(performance_score)
        
        # Manter apenas histórico recente
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da rede"""
        return {
            'current_size': self.current_size,
            'initial_size': self.initial_size,
            'max_size': self.max_size,
            'growth_ratio': self.current_size / self.initial_size,
            'avg_neuron_activation': self.neuron_activations.mean().item(),
            'neuron_usage_variance': self.neuron_activations.var().item(),
            'total_growth_events': len([h for h in self.growth_history if h['action'] == 'grow']),
            'total_prune_events': len([h for h in self.growth_history if h['action'] == 'prune']),
            'recent_performance': self._get_recent_performance()
        }
    
    def force_grow(self, amount: int = 1):
        """Força crescimento da rede (para debugging/testes)"""
        if self.current_size + amount <= self.max_size:
            old_size = self.current_size
            self.current_size = min(self.current_size + amount, self.max_size)
            self._grow_network()
            logger.info(f"Crescimento forçado: {old_size} -> {self.current_size}")
    
    def reset_stats(self):
        """Reseta estatísticas de uso"""
        self.neuron_activations.zero_()
        self.neuron_gradients.zero_()
        self.neuron_usage_count.zero_()
        self.step_count.zero_()
        self.performance_history.clear()
        self.growth_history.clear()

class AdaptiveTransformerLayer(nn.Module):
    """
    Camada Transformer com capacidade de crescimento
    """
    def __init__(
        self,
        hidden_size: int,
        max_hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_hidden_size = max_hidden_size
        self.num_heads = num_heads
        
        # Atenção multi-cabeça
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward crescente
        self.feed_forward = GrowingNetwork(
            initial_size=hidden_size,
            max_size=max_hidden_size,
            growth_rate=0.1,
            prune_threshold=0.01
        )
        
        # Normalizações
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass da camada adaptativa"""
        # Atenção com conexão residual
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward crescente com conexão residual
        # Ajustar dimensões se necessário
        if x.size(-1) != self.feed_forward.current_size:
            # Projetar para o tamanho atual da rede crescente
            if x.size(-1) < self.feed_forward.current_size:
                # Expandir
                padding = torch.zeros(*x.shape[:-1], 
                                    self.feed_forward.current_size - x.size(-1),
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=-1)
            else:
                # Truncar
                x = x[..., :self.feed_forward.current_size]
        
        ff_output = self.feed_forward(x)
        
        # Ajustar dimensões para conexão residual
        if ff_output.size(-1) != self.hidden_size:
            if ff_output.size(-1) > self.hidden_size:
                ff_output = ff_output[..., :self.hidden_size]
            else:
                padding = torch.zeros(*ff_output.shape[:-1], 
                                    self.hidden_size - ff_output.size(-1),
                                    device=ff_output.device, dtype=ff_output.dtype)
                ff_output = torch.cat([ff_output, padding], dim=-1)
        
        x_residual = x[..., :self.hidden_size]  # Garantir compatibilidade
        x = self.norm2(x_residual + self.dropout(ff_output))
        
        return x
    
    def update_performance(self, performance_score: float):
        """Atualiza performance da camada"""
        self.feed_forward.update_performance(performance_score)
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da camada"""
        return self.feed_forward.get_network_stats()

class StateSpaceLayer(nn.Module):
    """
    Camada State-Space para processamento eficiente de sequências longas.
    """
    def __init__(self, hidden_size: int, state_size: int = 16, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size

        self.A = nn.Parameter(torch.randn(state_size, state_size) * 0.1)
        self.B = nn.Parameter(torch.randn(state_size, hidden_size) * 0.1)
        self.C = nn.Parameter(torch.randn(hidden_size, state_size) * 0.1)
        self.D = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.cached_state = None
        self.has_cached_state = False

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if use_cache and self.has_cached_state:
            state = self.cached_state
        else:
            state = torch.zeros(batch_size, self.state_size, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            state = torch.matmul(state, self.A.T) + torch.matmul(x[:, t], self.B.T)
            output = torch.matmul(state, self.C.T) + torch.matmul(x[:, t], self.D.T)
            outputs.append(output)
        output = torch.stack(outputs, dim=1)
        if use_cache:
            self.cached_state = state.detach()
            self.has_cached_state = True
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output

    def parallel_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, use_cache=False)

    def clear_cache(self):
        self.cached_state = None
        self.has_cached_state = False