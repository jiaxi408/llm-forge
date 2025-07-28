from transformers import PretrainedConfig

# Yuanmeng model's configuration
class YuanMengConfig(PretrainedConfig):
    model_type = "yuanmeng"

    def __init__(
            self,
            hidden_size: int = 512,
            num_layers: int = 8,
            num_attention_heads: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            ffn_hidden_size: int = None,
            norm_eps: float = 1e-5,
            max_seq_len: int = 32 * 1024,
            rope_theta: int = 1e6,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.ffn_hidden_size = ffn_hidden_size
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        super().__init__(**kwargs)