import torch
from torch import nn

from typing import Optional, Tuple

import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoTokenizer

from .config import YuanMengConfig

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

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor):
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = self.weight * norm.type_as(x)
        return output
    
class YuanMengAttention(nn.Module):
    def __init__(self, config: YuanMengConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads
        self.head_size = config.hidden_size // self.num_attention_heads
        self.wq = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_size, bias=False, dtype=dtype, device=device)
        self.wk = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_size, bias=False, dtype=dtype, device=device)
        self.wv = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_size, bias=False, dtype=dtype, device=device)
        self.wo = nn.Linear(self.num_attention_heads * self.head_size, config.hidden_size, bias=False, dtype=dtype, device=device)
    
    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.num_attention_heads, self.head_size)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_size)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_size)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # kv chache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2), # (batch_size, num_attention_heads, seq_len, head_size)
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )


        if seq_len != 1:
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        else:
            # Streaming generation: disable causal masking so each new token can attend to all past keys
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=False)
        
        output = self.wo(output.transpose(1, 2).view(bsz, seq_len, -1))
        return output, past_kv

class YuanMengFeedForward(nn.Module):
    def __init__(self, config: YuanMengConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        if config.ffn_hidden_size is None:
            ffn_hidden_size = 4 * config.hidden_size # The original FFN hidden size is 4 times the model hidden size.
            ffn_hidden_size = int(2 * ffn_hidden_size / 3) # Maintain the same parameter count as the original FFN.
            config.ffn_hidden_size = 64 * ((ffn_hidden_size + 63) // 64) # Round up to the nearest multiple of 64
        self.w1 = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False, dtype=dtype, device=device)
        self.w3 = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class YuanMengDecoderLayer(nn.Module):
    def __init__(self, layer_id: int, config: YuanMengConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.layer_id = layer_id
        self.attention = YuanMengAttention(config, dtype=dtype, device=device)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)
        self.feed_forward = YuanMengFeedForward(config, dtype=dtype, device=device)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv
    
class YuanMengModel(PreTrainedModel):
    config_class = YuanMengConfig

    def __init__(self, config: YuanMengConfig = None, dtype: torch.dtype = None, device: torch.device = None):
        self.config = config or YuanMengConfig()
        super().__init__(self.config)
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size, dtype=dtype, device=device)
        self.layers = nn.ModuleList([YuanMengDecoderLayer(l, self.config, dtype=dtype, device=device) for l in range(self.config.num_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.norm_eps, dtype=dtype, device=device)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False, dtype=dtype, device=device)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer("pos_cis", 
                             precompute_pos_cis(head_size=self.config.hidden_size // self.config.num_attention_heads,
                                                seq_len=self.config.max_seq_len, theta=self.config.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: torch.Tensor,
                past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                start_pos: int = 0):
        h = self.tok_embeddings(input_ids)
        pos_cis = self.pos_cis[start_pos: start_pos + input_ids.size(1)]
        past_key_values = past_key_values or [None] * len(self.layers)
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(h, pos_cis, past_key_value=past_key_values[l], use_cache=use_cache)
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))

        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
    
    # mode = 0 indicates the chat mode of a pre-trained model, where no chat template is applied
    # mode = 1 indicates the chat mode of an SFT model, where a chat template is applied
    # When mode is set to 0, messages should be a string.
    # When mode is set to 1, it should be a list of dictionaries in the format [{"role": "...", "content": "..."}, ...]
    @torch.inference_mode()
    def chat(self, messages, mode = 1, stream = False, max_new_tokens = 1024, temperature = 0.75, top_p = 0.90):
        if stream:
            return self.stream(messages, mode, max_new_tokens, temperature, top_p)
        
        generated_text = ""
        generated_text_ids = []
        for item in self.stream(messages, mode, max_new_tokens, temperature, top_p):
            generated_text += item["text"]
            generated_text_ids.append(item["text_ids"])
            input_tokens_len = item["input_tokens_len"]
        
        return {
            "generated_text": generated_text,
            "generated_text_ids": torch.tensor(generated_text_ids),
            "genetated_tokens_len": len(generated_text_ids),
            "input_tokens_len": input_tokens_len
        }

    def stream(self, messages, mode, max_new_tokens, temperature, top_p):
        tokenizer = AutoTokenizer.from_pretrained("model/yuanmeng_tokenizer")
        if mode == 0:
            messages = tokenizer.bos_token + '<|im_start|>' + messages
            input_ids = tokenizer(messages, return_tensors='pt').input_ids
            input_ids.to(next(self.parameters()).device)
        elif mode == 1:
            input_ids = tokenizer.apply_chat_template(messages)
            input_ids = torch.tensor([input_ids], device=next(self.parameters()).device)

        if (self.config.max_seq_len - input_ids.shape[1]) <= 0:
            print(f"\033[91mError: The model's maximum context length is {self.config.max_seq_len} tokens, but the current conversation has {input_ids.shape[1]} tokens, which exceeds the limit.\033[0m")
            max_new_tokens = 0
            yield {"errors": "Context length overflow!"}
        elif (self.config.max_seq_len - input_ids.shape[1]) < max_new_tokens:
            print(f"\033[33mWarning: Due to the maximum context length limit of {self.config.max_seq_len} tokens, the model can generate at most {self.config.max_seq_len - input_ids.shape[1]} tokens in this response.\033[0m")
            max_new_tokens = self.config.max_seq_len - input_ids.shape[1]

        start, first_seq, past_kvs, text_next_ids, generated_text_ids = input_ids.shape[1], True, None, [], []
        while len(generated_text_ids) < max_new_tokens:
            if first_seq:
                out, first_seq = self(input_ids, use_cache=True), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=True,
                           start_pos=input_ids.shape[1] - 1)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values

            logits /= (temperature + 1e-9)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)

            text_next_ids.append(input_ids_next.squeeze().tolist())
            generated_text_ids.append(input_ids_next.squeeze().tolist())

            text_next = tokenizer.decode(text_next_ids, skip_special_tokens=True)
            generated_text = tokenizer.decode(generated_text_ids, skip_special_tokens=True)
            if generated_text[-1] == '�':
                continue

            yield {
                "text": text_next,
                "generated_text": generated_text,
                "text_ids": torch.tensor(text_next_ids),
                "generated_text_ids": torch.tensor(generated_text_ids),
                "genetated_tokens_len": len(generated_text_ids),
                "input_tokens_len": start
            }
            text_next_ids = []
            if input_ids_next.item() == tokenizer.eos_token_id:
                break