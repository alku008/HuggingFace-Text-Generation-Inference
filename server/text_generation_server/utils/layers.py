import torch

from torch import nn
from torch.nn import functional as F
from typing import Optional, List

HAS_BITS_AND_BYTES = True
try:
    from bitsandbytes.nn import Linear8bitLt
except ImportError as e:
    HAS_BITS_AND_BYTES = False


class FastLinear(nn.Module):
    def __init__(
        self,
        weight, bias,
        ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.T)
        if bias:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.addmm(self.bias, input, self.weight)
        return torch.matmul(input, self.weight)


def get_linear(weight, bias, quantize):
    if quantize is None:
        linear = FastLinear(weight, bias)
    elif quantize == "bitsandbytes":
        out_features, in_features = weight.shape
        linear = Linear8bitLt(
            in_features,
            out_features,
            has_fp16_weights=False,
            threshold=6.0,
            bias=bias,
        )
        linear.weight = nn.Parameter(weight)
        if bias:
            linear.bias = nn.Parameter(bias)
    elif quantize == "gptq":
        raise NotImplementedError("Soon")
    else:
        raise NotImplementedError(f"Quantization `{config.quantize}` is not implemented yet.")
    return linear


class SuperLayer(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear.forward(x)

class TensorParallelHead(SuperLayer):
    @staticmethod
    def load(config, prefix: str, weights, bias: bool):
        weight = weights.get_sharded(f"{prefix}.weight", dim=0) 
        if bias:
            bias = weights.get_sharded(f"{prefix}.bias", dim=0) 
        else:
            bias = None
        model = TensorParallelHead(get_linear(weight, bias, config.quantize))
        model.process_group = weights.process_group
        model.world_size = weights.process_group.size()
        return model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        # Logits are sharded, so we need to gather them
        world_output = [torch.empty_like(output) for _ in range(self.world_size)]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=1)
        return world_output


class TensorParallelColumnLinear(SuperLayer):
    @staticmethod
    def load(config, prefix: str, weights, bias: bool):
        weight = weights.get_sharded(f"{prefix}.weight", dim=0) 
        if bias:
            bias = weights.get_sharded(f"{prefix}.bias", dim=0) 
        else:
            bias = None
        return TensorParallelColumnLinear(get_linear(weight, bias, config.quantize))

    @staticmethod
    def load_multi(config, prefixes: List[str], weights, bias: bool, dim: int):
        w = [weights.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
        weight = torch.cat(w, dim=dim)

        if bias:
            b = [weights.get_sharded(f"{p}.bias", dim=0) for p in prefixes]
            bias = torch.cat(b, dim=dim)
        else:
            bias = None
        return TensorParallelColumnLinear(get_linear(weight, bias, config.quantize))


class TensorParallelRowLinear(SuperLayer):
    @staticmethod
    def load(config, prefix: str, weights, bias: bool):
        weight = weights.get_sharded(f"{prefix}.weight", dim=1) 
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias") 
        else:
            bias = None
        layer =  TensorParallelRowLinear(get_linear(weight, bias, config.quantize))
        layer.process_group = weights.process_group
        return layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        torch.distributed.all_reduce(out, group=self.process_group)
        return out

class TensorParallelEmbedding(nn.Module):
    def __init__(self, prefix: str, weights):
        super().__init__()
        weight = weights.get_sharded(f"{prefix}.weight", dim=0) 
        num_embeddings = weights.get_shape(f"{prefix}.weight")[0]

        process_group = weights.process_group

        world_size = process_group.size()
        rank = process_group.rank()

        block_size = num_embeddings // world_size
        assert num_embeddings % world_size == 0
        self.min_id = rank * block_size
        self.max_id = (rank + 1) * block_size
        self.null_idx = block_size
        self.process_group = weights.process_group

        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(weight, (0, 0, 0, 1)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        input = torch.where(
            (self.min_id > input) | (input >= self.max_id),
            self.null_idx,
            input - self.min_id,
        )
        out = torch.nn.functional.embedding(input, self.weight)
        # TODO self.reduce
        torch.distributed.all_reduce(out, group=self.process_group)
        return out


try:
    import dropout_layer_norm

    class FastLayerNorm(nn.LayerNorm):
        def forward(self, hidden_states, residual=None):
            if hidden_states.shape[-1] > 8192:
                if residual is not None:
                    hidden_states += residual
                residual = hidden_states

                return super(FastLayerNorm, self).forward(hidden_states), residual
            else:
                (
                    normed_hidden_states,
                    residual,
                    *rest,
                ) = dropout_layer_norm.dropout_add_ln_fwd(
                    hidden_states,
                    residual,
                    self.weight,
                    self.bias,
                    None,
                    None,
                    None,
                    None,
                    0.0,
                    self.eps,
                    1.0,
                    0,
                    None,
                    False,
                    False,
                )
                if residual is None:
                    residual = hidden_states

                return normed_hidden_states, residual

except ImportError:
    pass


try:
    from flash_attn.layers.rotary import RotaryEmbedding
    import rotary_emb

    class PositionRotaryEmbedding(RotaryEmbedding):
        def _update_cos_sin_cache(self, dtype, device, seqlen):
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
            ):
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                freqs = torch.outer(t, self.inv_freq.to(device=t.device))
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)

        def get_cos_sin(
            self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype
        ):
            """
            Return cos and sin for the asked position ids
            """

            self._update_cos_sin_cache(dtype, position_ids.device, max_s)

            cos = torch.index_select(self._cos_cached, 0, position_ids)
            sin = torch.index_select(self._sin_cached, 0, position_ids)
            return cos.unsqueeze(1), sin.unsqueeze(1)

        def forward(self, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            rotary_dim = cos.shape[-1]
            q1 = qkv[:, 0, :, :rotary_dim]
            q2 = qkv[:, 0, :, rotary_dim : 2 * rotary_dim]
            k1 = qkv[:, 1, :, :rotary_dim]
            k2 = qkv[:, 1, :, rotary_dim : 2 * rotary_dim]

            rotary_emb.apply_rotary(q1, q2, cos, sin, q1, q2, False)
            rotary_emb.apply_rotary(k1, k2, cos, sin, k1, k2, False)
            return qkv

except ImportError:
    pass
