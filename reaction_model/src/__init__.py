# Modern Transformer for Reaction Prediction

# Lazy imports to avoid requiring torch for simple operations like tokenizer training
def __getattr__(name):
    if name == "ModelConfig":
        from .config import ModelConfig
        return ModelConfig
    elif name == "TrainConfig":
        from .config import TrainConfig
        return TrainConfig
    elif name == "DataConfig":
        from .config import DataConfig
        return DataConfig
    elif name == "SMILESTokenizer":
        from .tokenizer import SMILESTokenizer
        return SMILESTokenizer
    elif name == "Transformer":
        from .transformer import Transformer
        return Transformer
    elif name == "CausalTransformer":
        from .decoder_model import CausalTransformer
        return CausalTransformer
    elif name == "RxnPredictor":
        from .predictor import RxnPredictor
        return RxnPredictor
    elif name == "load_predictor":
        from .predictor import load_predictor
        return load_predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
    "SMILESTokenizer",
    "Transformer",
    "CausalTransformer",
    "RxnPredictor",
    "load_predictor",
]
