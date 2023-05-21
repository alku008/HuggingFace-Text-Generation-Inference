# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_neox import GPTNeoXConfig
from typing import Optional

# Flash attention imports
import flash_attn_cuda

from text_generation_server.utils.layers import (
    FastLinear,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelHead,
    FastLayerNorm,
    PositionRotaryEmbedding,
    get_linear,
)

def load_row(config, prefix: str, weights, bias: bool):
    weight = weights.get_sharded(f"{prefix}.weight", dim=1) 
    if bias and weights.process_group.rank() == 0:
        # Rank is only on the first rank process
        bias = weights.get_tensor(f"{prefix}.bias") 
    else:
        bias = None

    linear = get_linear(weight, bias, config.quantize)
    if config.use_parallel_residual:
        return linear
    else:
        return TensorParallelColumnLinear(linear)

def load_col(config, prefix: str, weights, bias: bool):
    weight = weights.get_sharded(f"{prefix}.weight", dim=0) 
    if bias:
        bias = weights.get_sharded(f"{prefix}.bias", dim=0) 
    else:
        bias = None
    linear = get_linear(weight, bias, config.quantize)
    if config.use_parallel_residual:
        return linear
    else:
        return TensorParallelRowLinear(linear)

class FlashNeoxAttention(torch.nn.Module):
    def __init__(
        self,
        config, prefix, weights
    ):
        super().__init__()
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        rotary_pct = config.rotary_pct
        rotary_emb_base = config.rotary_emb_base

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.num_heads = self.num_heads // weights.process_group.size()

        rotary_ndims = int(self.head_size * rotary_pct)
        self.rotary_emb = PositionRotaryEmbedding(rotary_ndims, base=rotary_emb_base)
        self.softmax_scale = self.head_size ** (-0.5)

        self.query_key_value = load_col(
            config,prefix=f"{prefix}.query_key_value", weights=weights, bias=True
        )
        self.dense = load_row(
            config,prefix=f"{prefix}.dense", weights=weights, bias=True
        )

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_size)
        qkv_rot = self.rotary_emb(qkv, cos, sin)

        # Prefill
        if layer_past_present_indices is None:
            # Copy to layer past
            layer_past[...] = qkv_rot[:, 1:]

            # output
            attn_output = torch.empty_like(qkv_rot[:, 0])
            # flash attention
            flash_attn_cuda.fwd(
                qkv_rot[:, 0],
                qkv_rot[:, 1],
                qkv_rot[:, 2],
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
            query = qkv_rot[:, 0]
            # Add present to the layer_past tensor at the correct indices
            layer_past[layer_past_present_indices] = qkv_rot[:, 1:]

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                layer_past[:, 0],
                layer_past[:, 1],
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

        return self.dense(attn_output.view(-1, self.num_heads * self.head_size))


class FlashMLP(nn.Module):
    def __init__(
        self, config, prefix, weights
    ):
        super().__init__()
        act = config.hidden_act
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

        self.dense_h_to_4h = TensorParallelColumnLinear.load(
                config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=True
        )
        self.dense_4h_to_h = TensorParallelRowLinear.load(
            config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=True
        )

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class FlashNeoXLayer(nn.Module):
    def __init__(
        self,
        layer_id,
        config,
        weights
        # num_heads,
        # act,
        # hidden_size,
        # intermediate_size,
        # rotary_pct,
        # rotary_emb_base,
        # layer_norm_eps,
        # use_parallel_residual,
        # process_group=None,
    ):
        super().__init__()

        use_parallel_residual = config.use_parallel_residual
        hidden_size = config.hidden_size
        layer_norm_eps = config.layer_norm_eps

        prefix = f"gpt_neox.layers.{layer_id}"

        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = FastLayerNorm.load(prefix=f"{prefix}.input_layernorm", weights=weights, eps=layer_norm_eps)
        self.post_attention_layernorm = FastLayerNorm.load(prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=layer_norm_eps)
        self.attention = FlashNeoxAttention(
                config, prefix=f"{prefix}.attention", weights=weights
            # num_heads,
            # hidden_size,
            # rotary_pct,
            # rotary_emb_base,
            # process_group,
            # reduce=not use_parallel_residual,
        )
        self.mlp = FlashMLP(
                config, prefix=f"{prefix}.mlp", weights=weights
            # act,
            # hidden_size,
            # intermediate_size,
            # process_group,
            # reduce=not use_parallel_residual,
        )
        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        if self.use_parallel_residual:
            ln1_hidden_states, _ = self.input_layernorm(hidden_states)

            attn_output = self.attention(
                ln1_hidden_states,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past,
                layer_past_present_indices,
                cu_seqlens_q,
            )

            ln2_hidden_states, _ = self.post_attention_layernorm(hidden_states)

            mlp_output = self.mlp(ln2_hidden_states)
            intermediate = mlp_output + attn_output

            # Only reduce once and after the addition instead of once per layer
            if self.process_group is not None:
                torch.distributed.all_reduce(intermediate, group=self.process_group)

            return intermediate + hidden_states, None
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.attention(
                hidden_states,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past,
                layer_past_present_indices,
                cu_seqlens_q,
            )

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

            mlp_output = self.mlp(hidden_states)

            return mlp_output, residual


class FlashGPTNeoXPreTrainedModel(PreTrainedModel):
    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = False
    _no_split_modules = None


class FlashGPTNeoXModel(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.config = config

        self.embed_in = TensorParallelEmbedding(
            prefix="gpt_neox.embed_in", weights=weights
        )

        self.layers = nn.ModuleList(
            [
                FlashNeoXLayer(
                    layer_id,
                    config,
                    weights
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = FastLayerNorm.load(prefix="gpt_neox.final_layer_norm", weights=weights, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].attention.head_size
        self.num_heads = self.layers[0].attention.num_heads

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        past_key_values=None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        hidden_states = self.embed_in(input_ids)

        # Prefill
        if past_key_values is None:
            # Create past tensor
            past_key_values = hidden_states.new_empty(
                (
                    len(self.layers),
                    len(hidden_states)
                    if pre_allocate_past_size is None
                    else pre_allocate_past_size,
                    2,
                    self.num_heads,
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

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].attention.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            # We added padding that we now need to slice
            layer_past_key_values = (
                past_key_values[i]
                if slice_past_index is None
                else past_key_values[i, :slice_past_index]
            )

            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past_key_values,
                layer_past_present_indices,
                cu_seqlens_q,
            )

        hidden_states, _ = self.final_layer_norm(hidden_states, residual)

        return hidden_states, past_key_values


class FlashGPTNeoXForCausalLM(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.gpt_neox = FlashGPTNeoXModel(config, weights)

        self.embed_out = TensorParallelHead.load(config, prefix="embed_out", weights=weights)

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
        hidden_states, present = self.gpt_neox(
            input_ids,
            position_ids,
            cu_seqlens,
            cu_seqlens_q,
            max_s,
            past_key_values,
            pre_allocate_past_size,
        )
        logits = self.embed_out(hidden_states)
        return logits, present
