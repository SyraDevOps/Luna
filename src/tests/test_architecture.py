import unittest
import torch
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

from src.models.moe import MoEBlock
from src.models.hypernet import HyperNetwork, HyperLinear
from src.models.growing_network import GrowingNetwork, StateSpaceLayer
from src.models.luna_model import LunaModel
from src.config.config import Config, ModelConfig

class TestModelArchitecture(unittest.TestCase):
    """Testes para os componentes arquiteturais do modelo"""
    
    def setUp(self):
        """Configuração comum para os testes"""
        self.config = Config()
        self.batch_size = 2
        self.seq_len = 10
        self.input_dim = 64
        
    def test_moe_block(self):
        """Teste do bloco Mixture of Experts (MoE)"""
        # Configuração
        num_experts = 4
        moe = MoEBlock(
            input_dim=self.input_dim, 
            num_experts=num_experts,
            sparse_top_k=2
        )
        
        # Entrada de teste
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Testar sem roteamento emocional
        output = moe(x)
        self.assertEqual(output.shape, x.shape, "A forma da saída do MoE deve corresponder à entrada")
        
        # Testar com roteamento emocional
        emotion_weights = torch.softmax(torch.randn(self.batch_size, moe.emotional_dim), dim=-1)
        output_with_emotion = moe(x, emotion_weights)
        self.assertEqual(output_with_emotion.shape, x.shape, 
                         "A saída com pesos emocionais deve ter a mesma forma")
        
        # Verificar que o gradiente flui corretamente
        x.requires_grad = True
        output = moe(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
    
    def test_hyper_network(self):
        """Teste da HyperNetwork"""
        context_dim = 32
        target_dim = 128
        
        # Criar rede
        hypernet = HyperNetwork(context_dim=context_dim, target_dim=target_dim)
        
        # Testar geração de parâmetros
        context = torch.randn(self.batch_size, context_dim)
        weight, bias = hypernet(context)
        
        # Verificar formas
        self.assertEqual(weight.shape[0], self.batch_size)
        self.assertEqual(bias.shape[0], self.batch_size)
        
        # Verificar gradientes
        context.requires_grad = True
        weight, bias = hypernet(context)
        loss = (weight.sum() + bias.sum())
        loss.backward()
        self.assertIsNotNone(context.grad)
    
    def test_hyper_linear(self):
        """Teste da camada HyperLinear"""
        input_dim = 64
        output_dim = 32
        context_dim = 16
        
        # Criar camada
        hyper_linear = HyperLinear(input_dim, output_dim, context_dim)
        
        # Testar forward pass
        x = torch.randn(self.batch_size, self.seq_len, input_dim)
        context = torch.randn(self.batch_size, context_dim)
        
        output = hyper_linear(x, context)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, output_dim))
        
        # Verificar gradientes
        x.requires_grad = True
        context.requires_grad = True
        output = hyper_linear(x, context)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(context.grad)
    
    def test_growing_network(self):
        """Teste da GrowingNetwork"""
        # Modelo base simples para teste
        base_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.input_dim),
            torch.nn.ReLU()
        )
        
        # Criar rede que pode crescer
        growing_net = GrowingNetwork(base_model)
        
        # Testar forward com o modelo base
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output_base = growing_net(x)
        self.assertEqual(output_base.shape, x.shape)
        
        # Adicionar camada e testar novamente
        growing_net.add_layer(self.input_dim, self.input_dim)
        output_after_growth = growing_net(x)
        self.assertEqual(output_after_growth.shape, x.shape)
        
        # Verificar que as saídas são diferentes após a adição da camada
        self.assertFalse(torch.allclose(output_base, output_after_growth))
        
        # Testar limite de crescimento
        growing_net.growth_config['max_additional_layers'] = 2
        growing_net.growth_count = 2
        result = growing_net.add_layer(self.input_dim, self.input_dim)
        self.assertFalse(result, "Deve retornar False quando o limite de camadas for atingido")
    
    def test_state_space_layer(self):
        """Teste da State Space Layer"""
        hidden_size = 64
        state_size = 16
        
        # Criar camada
        ssl = StateSpaceLayer(hidden_size=hidden_size, state_size=state_size)
        
        # Testar forward
        x = torch.randn(self.batch_size, self.seq_len, hidden_size)
        output = ssl(x)
        
        # Verificar forma da saída
        self.assertEqual(output.shape, x.shape)
        
        # Verificar gradientes
        x.requires_grad = True
        output = ssl(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
    
    def test_luna_model_with_components(self):
        """Teste da integração dos componentes na LunaModel"""
        # Configuração para usar todos os componentes
        model_config = ModelConfig()
        model_config.use_moe = True
        model_config.use_state_space = True
        model_config.use_growing_network = True
        model_config.num_experts = 2  # Reduzir para teste
        
        # Criar modelo completo
        with patch('src.models.luna_model.detect_hardware') as mock_detect:
            # Simular hardware standard
            mock_hardware = MagicMock()
            mock_hardware.system_type = "standard"
            mock_detect.return_value = mock_hardware
            
            # Criar modelo
            model = LunaModel.from_scratch(model_config)
        
        # Verificar se os componentes foram criados
        self.assertIsNotNone(model.model, "Modelo base deve ser criado")
        self.assertIsNotNone(model.moe_blocks, "Blocos MoE devem ser criados")
        
        # Salvar e carregar modelo com componentes
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Salvar
            model.save(tmpdirname)
            
            # Verificar componentes salvos
            components_dir = os.path.join(tmpdirname, "components")
            self.assertTrue(os.path.exists(components_dir))
            
            # Carregar usando from_pretrained
            loaded_model = LunaModel.from_pretrained(tmpdirname, model_config)
            self.assertIsNotNone(loaded_model.model)

if __name__ == '__main__':
    unittest.main()