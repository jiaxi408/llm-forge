# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

# Maps that describe the structure of your model.
NORM_FCS_MAP = {
    'YuanMengDecoderLayer': {
        'attention_norm': ['attention.wq', 'attention.wk', 'attention.wv'],
        'ffn_norm': ['feed_forward.w1', 'feed_forward.w3']
    },
    'LlamaDecoderLayer': {
        'input_layernorm': ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
    'InternLMDecoderLayer': {
        'input_layernorm': ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
    'InternLM2DecoderLayer': {
        'attention_norm': ['attention.wqkv'],
        'ffn_norm': ['feed_forward.w1', 'feed_forward.w3']
    },
    'InternLM3DecoderLayer': {
        'input_layernorm': ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
    'QWenBlock': {
        'ln_1': ['attn.c_attn'],
        'ln_2': ['mlp.w1', 'mlp.w2']
    },
    'Qwen2DecoderLayer': {
        'input_layernorm': ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
    'Qwen3DecoderLayer': {
        'input_layernorm': ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
    'DecoderLayer': {
        'input_layernorm': ['self_attn.W_pack'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
    'Phi3DecoderLayer': {
        'input_layernorm': ['self_attn.qkv_proj'],
        'post_attention_layernorm': ['mlp.gate_up_proj']
    },
    'GLMBlock': {
        'input_layernorm': ['self_attention.query_key_value'],
        'post_attention_layernorm': ['mlp.dense_h_to_4h']
    },
    'MixtralDecoderLayer': {
        'input_layernorm': ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm':
        ['block_sparse_moe.gate', 'block_sparse_moe.experts.{i}.w1', 'block_sparse_moe.experts.{i}.w3']
    },
    'Qwen2VLDecoderLayer': {
        'input_layernorm': ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
    'Qwen2_5_VLDecoderLayer': {
        'input_layernorm': ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
    'MistralDecoderLayer': {
        'input_layernorm': ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
}

FC_FCS_MAP = {
    'YuanMengDecoderLayer': {
        'attention.wv': ['attention.wo'],
        'feed_forward.w3': ['feed_forward.w2']
    },
    'LlamaDecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    },
    'InternLMDecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    },
    'InternLM2DecoderLayer': {
        'feed_forward.w3': ['feed_forward.w2']
    },
    'InternLM3DecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    },
    'QWenBlock': {
        'attn.c_attn': ['attn.c_proj'],
        'mlp.w1': ['mlp.c_proj']
    },
    'Qwen2DecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    },
    'Qwen3DecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    },
    'DecoderLayer': {
        'self_attn.W_pack': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    },
    'Phi3DecoderLayer': {
        'self_attn.qkv_proj': ['self_attn.o_proj'],
        'mlp.gate_up_proj': ['mlp.down_proj']
    },
    'GLMBlock': {
        # 'self_attention.query_key_value': ['self_attention.dense']
        # 'mlp.dense_h_to_4h': ['mlp.dense_4h_to_h']
    },
    'MixtralDecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'block_sparse_moe.experts.{i}.w3': ['block_sparse_moe.experts.{i}.w2']
    },
    'Qwen2VLDecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    },
    'Qwen2_5_VLDecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    },
    'MistralDecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    }
}

SKIPPED_MODULE = ['lora', 'block_sparse_moe.gate']


def skipped_module(name: str):
    """Whether the module should be skipped from quantization."""
    for m in SKIPPED_MODULE:
        if m in name:
            return True
    return False


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    abs_weight = weight.abs()
    abs_weight_amax = abs_weight.amax(dim=1, keepdim=True)
    if abs_weight_amax.min().item() == 0:
        print('weight.amax.min is zero, clamping weight.amax to 1e-4')
        abs_weight_amax = abs_weight_amax.clamp(min=1e-4)
    scale = abs_weight / abs_weight_amax
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def smooth_ln_fcs(ln: torch.nn.Module,
                  fcs: List[torch.nn.Module],
                  act_scales: torch.Tensor,
                  group_size: int = -1,
                  alpha: float = 0.5) -> torch.Tensor:
    """Smooth weights of a layer normalization and its fully connected layers.

    :param ln: Layer Normalization module
    :param fcs: List of Fully Connected modules
    :param act_scales: Activation scales
    :param alpha: Scaling factor (default is 0.5)
    :return: Scales
    """
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype

    # If zeros exist within the weight of the layer norm, it becomes
    # unnecessary to perform smooth quantization at the positions where
    # these zeros occur.
    zero_positions = (ln.weight == 0).nonzero(as_tuple=True)[0]
    nonzero_positions = (ln.weight != 0).nonzero(as_tuple=True)[0]

    act_scales = act_scales.to(device=device, dtype=dtype)

    concat_w = torch.cat([fc.weight for fc in fcs], dim=0)
    w_scales = get_weight_scale(concat_w, group_size)

    w_scales_pow = w_scales.pow(1 - alpha)
    if w_scales_pow.min().item() == 0:
        print('w_scales.pow(1 - alpha).min is zero, '
              'clamping w_scales.pow(1 - alpha) to 1e-4')
        w_scales_pow = w_scales_pow.clamp(min=1e-4)
    scales = (act_scales.pow(alpha) / w_scales_pow).clamp(min=1e-4).to(device).to(dtype)

    scales = scales / (scales[nonzero_positions].max() * scales[nonzero_positions].min()).sqrt()

    scales[zero_positions] = 1

    ln.weight.div_(scales)
    if hasattr(ln, 'bias'):
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0
    return scales


@torch.no_grad()
def smooth_fc_fcs(pre_fc: torch.nn.Module,
                  fcs: List[torch.nn.Module],
                  act_scales: torch.Tensor,
                  group_size: int = -1,
                  alpha: float = 0.5) -> torch.Tensor:
    """Smooth weights of a fully connected layer and its downstream layers.

    :param pre_fc: Previous Fully Connected layer
    :param fcs: List of Fully Connected modules
    :param act_scales: Activation scales
    :param alpha: Scaling factor (default is 0.5)
    :return: Scales
    """
    device, dtype = pre_fc.weight.device, pre_fc.weight.dtype

    size_a = act_scales.size(0)
    size_pre_fc = pre_fc.weight.size(0)

    # (for llama2) use group query attention, pre_fc is v_proj, fc is o_proj
    if size_pre_fc < size_a and size_a % size_pre_fc == 0:
        return

    act_scales = act_scales.to(device=device, dtype=dtype)

    concat_w = torch.cat([fc.weight for fc in fcs], dim=0)
    w_scales = get_weight_scale(concat_w, group_size)

    w_scales_pow = w_scales.pow(1 - alpha)
    if w_scales_pow.min().item() == 0:
        print('w_scales.pow(1 - alpha).min is zero, '
              'clamping w_scales.pow(1 - alpha) to 1e-4')
        w_scales_pow = w_scales_pow.clamp(min=1e-4)
    scales = (act_scales.pow(alpha) / w_scales_pow).clamp(min=1e-4).to(device).to(dtype)
    scales = scales / (scales.max() * scales.min()).sqrt()

    # (for qwen&baichuan) pre_fc is packed QKV, only V needs to scale
    # phi3 fused qkv and gate_up
    if size_pre_fc > size_a and size_pre_fc % size_a == 0 \
            and size_pre_fc // size_a in [2, 3]:

        pre_fc.weight[-size_a:].div_(scales.view(-1, 1))

        if getattr(pre_fc, 'bias', None) is not None:
            pre_fc.bias[-size_a:].div_(scales)
    else:

        pre_fc.weight.div_(scales.view(-1, 1))

        if getattr(pre_fc, 'bias', None) is not None:
            pre_fc.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in pre_fc.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

    return scales


def check_awq_supported(layer_type):
    """Check if the smooth function is supported by inspecting layer type."""
    norm_fcs_found = False
    fc_fcs_found = False

    if isinstance(layer_type, str):
        if layer_type in NORM_FCS_MAP:
            norm_fcs_found = True
        if layer_type in FC_FCS_MAP:
            fc_fcs_found = True

    elif isinstance(layer_type, type):
        if layer_type.__name__ in NORM_FCS_MAP:
            norm_fcs_found = True
        if layer_type.__name__ in FC_FCS_MAP:
            fc_fcs_found = True

    else:
        raise NotImplementedError

    if not norm_fcs_found:
        raise NotImplementedError

    if not fc_fcs_found:
        raise NotImplementedError


def quant_weights(model, fcs, bits, symmetry, group_size=-1, device='cuda'):
    """Quantize the weights of the target model's linear layers."""
    from lmdeploy.lite.quantization import WeightQuantizer
    from lmdeploy.lite.quantization.modules import WeightOnlyQLinear
    from lmdeploy.lite.utils import QParams
    for name, fc in fcs.items():
        fc.to(device)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        pack_or_skip = 'packed'
        if skipped_module(name):
            q_linear = fc
            pack_or_skip = 'skipped'
        else:
            quantizer = WeightQuantizer(bits, symmetry, 'per_group', group_size)
            fc.weight.data, scales, zeros = pseudo_quantize_tensor(fc.weight.data,
                                                                   bits,
                                                                   group_size,
                                                                   return_scale_zeros=True)
            q_linear = WeightOnlyQLinear.from_linear(fc, quantizer, qparams=QParams(scales, zeros))
        setattr(parent, child_name, q_linear)
        fc.to('cpu')
        torch.cuda.empty_cache()

        print(f'{name} weight {pack_or_skip}.')


def smooth_layers(layers, fc2fcs, norm2fcs, a_scales, group_size=-1, device='cuda'):
    """Apply weight smoothing based on input scales."""

    for l_name, layer in layers.items():
        layer.to(device)
        submodule_names = [name for name, _ in layer.named_modules()]
        for ln_name, fc_names in norm2fcs.items():
            a_name = [f'{l_name}.{n}' for n in fc_names if n in submodule_names][0]

            ln = layer.get_submodule(ln_name)
            fcs = [layer.get_submodule(n) for n in fc_names if n in submodule_names]
            smooth_ln_fcs(ln, fcs, a_scales[a_name], group_size)

        for f_name, fc_names in fc2fcs.items():
            a_name = [f'{l_name}.{n}' for n in fc_names if n in submodule_names][0]

            fc = layer.get_submodule(f_name)
            fcs = [layer.get_submodule(n) for n in fc_names if n in submodule_names]

            smooth_fc_fcs(fc, fcs, a_scales[a_name], group_size)

        layer.to('cpu')
        torch.cuda.empty_cache()
        max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f'{l_name} smooth weight done.'
              f' max gpu memory: {max_memory:.2f} GB')


def pseudo_quantize_tensor(w, w_bit=8, w_group_size=-1, return_scale_zeros=False):
    """Pseudo quantize tensor."""
    org_w_shape = w.shape
    if w_group_size > 0:
        assert org_w_shape[-1] % w_group_size == 0
        w = w.reshape(-1, w_group_size)
    assert w.dim() == 2
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**w_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    q_w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w = (q_w - zeros) * scales
    assert torch.isnan(w).sum() == 0

    if return_scale_zeros:
        zeros = zeros.view(org_w_shape[0], org_w_shape[-1] // w_group_size, -1)
        scales = scales.view(org_w_shape[0], org_w_shape[-1] // w_group_size, -1)
        q_w = q_w.reshape(org_w_shape)
        return q_w, scales, zeros
    w = w.reshape(org_w_shape)
    return w


def awq_layers(layers, fc2fcs, norm2fcs, a_scales, a_ratios=None, group_size=-1, device='cuda'):
    """Apply awq based on input scales."""

    for l_name, layer in layers.items():
        layer.to(device)
        for ln_name, fc_names in norm2fcs.items():
            a_name = [f'{l_name}.{n}' for n in fc_names][0]
            ratios = [a_ratios[f'{l_name}.{n}'] for n in fc_names]
            ratio = [s for s in ratios if s is not None][0]

            ln = layer.get_submodule(ln_name)
            fcs = [layer.get_submodule(n) for n in fc_names]
            smooth_ln_fcs(ln, fcs, a_scales[a_name], group_size, ratio)

        for f_name, fc_names in fc2fcs.items():
            a_name = [f'{l_name}.{n}' for n in fc_names][0]
            ratios = [a_ratios[f'{l_name}.{n}'] for n in fc_names]
            ratios = [s for s in ratios if s is not None]
            ratio = 0.5 if not len(ratios) else ratios[0]

            fc = layer.get_submodule(f_name)
            fcs = [layer.get_submodule(n) for n in fc_names]

            smooth_fc_fcs(fc, fcs, a_scales[a_name], group_size, ratio)

        layer.to('cpu')
        torch.cuda.empty_cache()
        max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f'{l_name} smooth weight done.'
              f' max gpu memory: {max_memory:.2f} GB')
