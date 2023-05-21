import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, List

# Flash attention imports
import flash_attn_cuda
from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelHead,
    TensorParallelEmbedding,
    FastLayerNorm,
    get_linear,
)


def load_multi_mqa(config, prefixes: List[str], weights, bias: bool):
    if config.transpose:
        w = [
            weights.get_sharded(f"{p}.weight", dim=1).T
            if i == 0
            else weights.get_tensor(f"{p}.weight").T
            for i, p in enumerate(prefixes)
        ]
        weight = torch.cat(w, dim=0)
    else:
        w = [
            weights.get_sharded(f"{p}.weight", dim=0)
            if i == 0
            else weights.get_tensor(f"{p}.weight")
            for i, p in enumerate(prefixes)
        ]
        weight = torch.cat(w, dim=1)

    if bias:
        b = [
            weights.get_sharded(f"{p}.bias", dim=0)
            if i == 0
            else weights.get_tensor(f"{p}.bias")
            for i, p in enumerate(prefixes)
        ]
        bias = torch.cat(b, dim=0)
    else:
        bias = None

    return TensorParallelColumnLinear(get_linear(weight, bias, config.quantize))


def load_col(config, prefix: str, weights, bias: bool):
    if config.transpose:
        weight = weights.get_sharded(f"{prefix}.weight", dim=1).T
    else:
        weight = weights.get_sharded(f"{prefix}.weight", dim=0)

    if bias:
        bias = weights.get_sharded(f"{prefix}.bias", dim=0)
    else:
        bias = None
    return TensorParallelColumnLinear(get_linear(weight, bias, config.quantize))


def load_row(config, prefix: str, weights, bias: bool):
    if config.transpose:
        weight = weights.get_sharded(f"{prefix}.weight", dim=0).T
    else:
        weight = weights.get_sharded(f"{prefix}.weight", dim=1)

    if bias and weights.process_group.rank() == 0:
        # Rank is only on the first rank process
        bias = weights.get_tensor(f"{prefix}.bias")
    else:
        bias = None
    layer = TensorParallelRowLinear(get_linear(weight, bias, config.quantize))
    layer.process_group = weights.process_group
    return layer


class FlashMQAttention(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.num_heads = self.num_heads // weights.process_group.size()

        self.softmax_scale = self.head_size ** (-0.5)

        self.c_attn = load_multi_mqa(
            config,
            prefixes=[f"{prefix}.q_attn", f"{prefix}.kv_attn"],
            bias=True,
            weights=weights,
        )
        self.c_proj = load_row(
            config, prefix=f"{prefix}.c_proj", weights=weights, bias=True
        )

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        qkv = self.c_attn(hidden_states)

        # Split query from key_value
        query, key_value = qkv.split(
            [self.head_size * self.num_heads, 2 * self.head_size], dim=1
        )

        # Prepare query and key_value for indexing
        query = query.view(-1, self.num_heads, self.head_size)
        key_value = key_value.view(-1, 2, 1, self.head_size)

        # Prefill
        if layer_past_present_indices is None:
            # Copy to layer past
            layer_past[...] = key_value
            # Expand from 1 to num_heads
            key_value = key_value.expand(-1, 2, self.num_heads, self.head_size)

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                key_value[:, 0],
                key_value[:, 1],
                attn_output,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                0.0,
                self.softmax_scale,
                False,
                True,
                False,
                0,
                None,
            )
        # Decode
        else:
            # Add present to the layer_past tensor at the correct indices
            layer_past[layer_past_present_indices] = key_value
            # Expand from 1 to num_heads
            key_value = layer_past.expand(-1, 2, self.num_heads, self.head_size)

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                key_value[:, 0],
                key_value[:, 1],
                attn_output,
                cu_seqlens_q,
                cu_seqlens,
                1,
                max_s,
                0.0,
                self.softmax_scale,
                False,
                False,
                False,
                0,
                None,
            )

        return self.c_proj(attn_output.view(-1, self.num_heads * self.head_size))


class MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.activation_function
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )

        self.c_fc = load_col(
            config, prefix=f"{prefix}.c_fc", weights=weights, bias=True
        )
        self.c_proj = load_row(
            config, prefix=f"{prefix}.c_proj", weights=weights, bias=True
        )

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class Block(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"transformer.h.{layer_id}"
        self.ln_1 = FastLayerNorm.load(
            prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
        )
        self.ln_2 = FastLayerNorm.load(
            prefix=f"{prefix}.ln_2", weights=weights, eps=config.layer_norm_epsilon
        )
        self.attn = FlashMQAttention(
            prefix=f"{prefix}.attn",
            config=config,
            weights=weights,
        )
        self.mlp = MLP(
            prefix=f"{prefix}.mlp",
            config=config,
            weights=weights,
        )

    def forward(
        self,
        hidden_states,
        residual,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        hidden_states, residual = self.ln_1(hidden_states, residual)

        hidden_states = self.attn(
            hidden_states,
            cu_seqlens,
            max_s,
            layer_past,
            layer_past_present_indices,
            cu_seqlens_q,
        )

        hidden_states, residual = self.ln_2(hidden_states, residual)

        mlp_output = self.mlp(hidden_states)

        return mlp_output, residual


class FlashSantacoderModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config

        self.process_group = weights.process_group
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.wte = TensorParallelEmbedding(
            prefix="transformer.wte",
            weights=weights,
            reduce=False,
        )
        self.wpe = TensorParallelEmbedding(
            prefix="transformer.wpe",
            weights=weights,
            reduce=False,
        )
        self.tp_embeddings = True

        self.h = nn.ModuleList(
            [
                Block(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = FastLayerNorm.load(
            prefix="transformer.ln_f", weights=weights, eps=config.layer_norm_epsilon
        )

        self.head_size = self.h[0].attn.head_size
        self.num_heads = self.h[0].attn.num_heads

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        if self.tp_embeddings:
            torch.distributed.all_reduce(hidden_states, group=self.process_group)

        # Prefill
        if past_key_values is None:
            # Create past tensor
            past_key_values = hidden_states.new_empty(
                (
                    len(self.h),
                    len(hidden_states)
                    if pre_allocate_past_size is None
                    else pre_allocate_past_size,
                    2,
                    1,
                    self.head_size,
                )
            )
            layer_past_present_indices = None
            slice_past_index = len(hidden_states)
        # Decode
        else:
            # Create indices from cumulative sequence lengths
            layer_past_present_indices = cu_seqlens[1:] - 1
            slice_past_index = None

        residual = None
        for i, layer in enumerate(self.h):
            # We added padding that we now need to slice
            layer_past_key_values = (
                past_key_values[i]
                if slice_past_index is None
                else past_key_values[i, :slice_past_index]
            )

            hidden_states, residual = layer(
                hidden_states,
                residual,
                cu_seqlens,
                max_s,
                layer_past_key_values,
                layer_past_present_indices,
                cu_seqlens_q,
            )

        hidden_states, _ = self.ln_f(hidden_states, residual)

        return hidden_states, past_key_values


class FlashSantacoderForCausalLM(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.transformer = FlashSantacoderModel(config, weights)
        self.lm_head = TensorParallelHead.load(
            config, prefix="transformer.wte", weights=weights
        )

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        hidden_states, present = self.transformer(
            input_ids,
            position_ids,
            cu_seqlens,
            cu_seqlens_q,
            max_s,
            past_key_values,
            pre_allocate_past_size,
        )
        logits = self.lm_head(hidden_states)
        return logits, present
