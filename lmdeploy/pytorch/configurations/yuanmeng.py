from lmdeploy.pytorch.config import ModelConfig
from .builder import AutoModelConfigBuilder

class YuanMengModelConfigBuilder(AutoModelConfigBuilder):
    
    @classmethod
    def condition(cls, hf_config):
        return hf_config.model_type in ['yuanmeng']
    
    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        return ModelConfig(hidden_size=hf_config.hidden_size,
                           num_layers=hf_config.num_layers,
                           num_attention_heads=hf_config.num_attention_heads,
                           num_key_value_heads=hf_config.num_key_value_heads,
                           bos_token_id=1,
                           eos_token_id=2,
                           head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
                           vocab_size=hf_config.vocab_size)