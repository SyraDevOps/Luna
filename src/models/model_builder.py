from src.models.luna_model import LunaModel
from src.config.config import Config

def build_model(config, multi_task=False):
    return LunaModel(config.model)  # Pass the ModelConfig object