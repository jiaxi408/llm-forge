import torch
from torch import nn

from typing import Any, Optional, Tuple, List, Iterable

import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.pytorch.nn.linear import (build_down_linear, build_gateup_linear, build_o_proj, build_qkv_proj)

from lmdeploy.pytorch.nn import Attention, RMSNorm, SiluAndMul
from .utils.cudagraph import CudaGraphMixin

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

# output: shape -> (seq_len, head_size/2), dtype -> complex64
def precompute_pos_cis(head_size: int, seq_len: int = 32 * 1024, theta: float = 1e6):
    freqs = 1 / (theta ** (torch.arange(0, head_size, 2).float() / head_size))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis

# input: xq.shape -> (batch_size, seq_len, num_attention_heads, head_size), 
#        xk.shape -> (batch_size, seq_len, num_kv_heads, head_size)
#        pos_cis.shape -> (seq_len, head_size/2)
# output: xq.shape -> (batch_size, seq_len, num_attention_heads, head_size)
#         xk.shape -> (batch_size, seq_len, num_kv_heads, head_size)
def apply_rotary_emb(xq, xk, pos_cis):
    # During lmdeploy quantization, xq and pos_cis might not be on the same device and need to be explicitly set.
    pos_cis = pos_cis.to(xq.device)

    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2)) # (batch_size, seq_len, num_attention_heads, head_size/2)
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2)) # (batch_size, seq_len, num_kv_heads, head_size/2)
    shape = [d if i == 1 or i == 3 else 1 for i, d in enumerate(xq_.shape)]
    pos_cis = pos_cis.view(*shape) # (1, seq_len, 1, head_size/2)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int):
    bsz, seq_len, num_key_value_heads, head_size = x.shape
    if n_rep == 1:
        return x
    return x.unsqueeze(3).expand(bsz, seq_len, num_key_value_heads, n_rep, head_size).reshape(bsz, seq_len, num_key_value_heads * n_rep, head_size)
    
class YuanMengAttention(nn.Module):
    def __init__(self, config: YuanMengConfig, dtype: torch.dtype = None, device: torch.device = None):
        quantization_config = getattr(config, 'quantization_config', None)
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads
        self.head_size = config.hidden_size // self.num_attention_heads
        self.w_qkv = build_qkv_proj(
            config.hidden_size,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_size=self.head_size,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device
        )
        self.wo = build_o_proj(config.num_attention_heads * self.head_size,
                                   config.hidden_size,
                                   bias=False,
                                   quant_config=quantization_config,
                                   dtype=dtype,
                                   device=device,
                                   is_tp=True)
        self.attn_fwd = Attention(
            config.num_attention_heads,
            self.head_size,
            num_kv_heads=self.num_key_value_heads,
            v_head_size=self.head_size,
        )
    
    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                attn_metadata: Any = None):
        x_qkv = self.w_qkv(x)
        xq, xk, xv = self.w_qkv.split_qkv(x_qkv)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        xq = xq.flatten(0, 1)
        xk = xk.flatten(0, 1)
        xv = xv.flatten(0, 1)

        output = self.attn_fwd(
            xq,
            xk,
            xv,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        
        output = output.reshape(*x.shape[:-1], -1)
        output = self.wo(output)
        return output

class YuanMengFeedForward(nn.Module):
    def __init__(self, config: YuanMengConfig, dtype: torch.dtype = None, device: torch.device = None):
        quantization_config = getattr(config, 'quantization_config', None)
        super().__init__()
        if config.ffn_hidden_size is None:
            ffn_hidden_size = 4 * config.hidden_size # The original FFN hidden size is 4 times the model hidden size.
            ffn_hidden_size = int(2 * ffn_hidden_size / 3) # Maintain the same parameter count as the original FFN.
            config.ffn_hidden_size = 64 * ((ffn_hidden_size + 63) // 64) # Round up to the nearest multiple of 64
        self.w_13 = build_gateup_linear(
            config.hidden_size,
            [config.ffn_hidden_size, config.ffn_hidden_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )
        self.w2 = build_down_linear(config.ffn_hidden_size,
                                           config.hidden_size,
                                           bias=False,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)
        self.act_fn = SiluAndMul(inplace=True)

    def forward(self, x: torch.Tensor):
        gate_up = self.w_13(x)
        act = self.act_fn(gate_up)
        return self.w2(act)
    
class YuanMengDecoderLayer(nn.Module):
    def __init__(self, layer_id: int, config: YuanMengConfig, dtype: torch.dtype = None, device: torch.device = None):
        quantization_config = getattr(config, 'quantization_config', None)
        super().__init__()
        self.layer_id = layer_id
        self.attention = YuanMengAttention(config, dtype=dtype, device=device)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, quant_config=quantization_config, dtype=dtype, device=device)
        self.feed_forward = YuanMengFeedForward(config, dtype=dtype, device=device)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, quant_config=quantization_config, dtype=dtype, device=device)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: List[List[torch.Tensor]] = None,
                attn_metadata: Any = None):
        h_attn = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
class YuanMengModel(PreTrainedModel, CudaGraphMixin):
    config_class = YuanMengConfig

    def __init__(self, config: YuanMengConfig = None, ctx_mgr: StepContextManager = None, dtype: torch.dtype = None, device: torch.device = None):
        quantization_config = getattr(config, 'quantization_config', None)
        self.ctx_mgr = ctx_mgr
        self.config = config or YuanMengConfig()
        super().__init__(self.config)
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size, dtype=dtype, device=device)
        self.layers = nn.ModuleList([YuanMengDecoderLayer(l, self.config, dtype=dtype, device=device) for l in range(self.config.num_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.norm_eps, quant_config=quantization_config, dtype=dtype, device=device)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False, dtype=dtype, device=device)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer("pos_cis", 
                             precompute_pos_cis(head_size=self.config.hidden_size // self.config.num_attention_heads,
                                                seq_len=self.config.max_seq_len, theta=self.config.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                position_ids: torch.Tensor = None,
                past_key_values: List[List[torch.Tensor]] = None,
                attn_metadata: Any = None,
                **args):
        h = self.tok_embeddings(input_ids)
        pos_cis = self.pos_cis[position_ids[0]]
        for l, layer in enumerate(self.layers):
            h = layer(h, pos_cis, past_key_value=past_key_values[l], attn_metadata=attn_metadata)
        return h
    
    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.output(self.norm(hidden_states))
    
    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        stacked_params_mapping = [
            ('.w_qkv', '.wq', 'q'),
            ('.w_qkv', '.wk', 'k'),
            ('.w_qkv', '.wv', 'v'),
            ('.w_13', '.w1', 0),
            ('.w_13', '.w3', 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                load_weight(param, loaded_weight)