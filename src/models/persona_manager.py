from dataclasses import dataclass
import torch
import torch.nn as nn

class EmotionalPersona(nn.Module):
    def __init__(self, traits: dict):
        super().__init__()
        self.traits = nn.ParameterDict({
            "empathy": nn.Parameter(torch.tensor(traits.get("empathy", 0.5))),
            "curiosity": nn.Parameter(torch.tensor(traits.get("curiosity", 0.3))),
        })

    def forward(self):
        keys = sorted(self.traits.keys())
        return torch.stack([self.traits[k] for k in keys])

class PersonaManager(nn.Module):
    def __init__(self, config: dict, embedding_dim: int):
        super().__init__()
        self.persona_dict = config['personas']
        self.persona_names = list(self.persona_dict.keys())
        self.embeddings = nn.Embedding(len(self.persona_names), embedding_dim)
        self.default = config['default']
        self.emotional_personas = {}
        for name in self.persona_names:
            traits = {
                "empathy": self.persona_dict[name].get("empathy", 0.5),
                "curiosity": self.persona_dict[name].get("curiosity", 0.3)
            }
            self.emotional_personas[name] = EmotionalPersona(traits)

    def get_persona_embedding(self, persona_name: str) -> torch.Tensor:
        if persona_name not in self.persona_names:
            persona_name = self.default
        idx = self.persona_names.index(persona_name)
        return self.embeddings(torch.tensor(idx))

    def get_emotional_embedding(self, persona_name: str) -> torch.Tensor:
        if persona_name not in self.persona_names:
            persona_name = self.default
        return self.emotional_personas[persona_name]()

    def get_temperature(self, persona_name: str) -> float:
        return self.persona_dict.get(persona_name, {}).get("temperature", 0.7)