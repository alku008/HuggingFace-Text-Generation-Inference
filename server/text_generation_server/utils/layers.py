import torch

from torch import nn
from loguru import logger
from torch.nn import functional as F
from typing import Optional, List

HAS_BITS_AND_BYTES = True
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Int8Params

except ImportError as e:
    HAS_BITS_AND_BYTES = False

from accelerate import init_empty_weights


# Monkey patching
@staticmethod
def load_layer_norm(prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    bias = weights.get_tensor(f"{prefix}.bias")
    with init_empty_weights():
        ln = torch.nn.LayerNorm(weight.shape, eps=eps)

    ln.weight = nn.Parameter(weight)
    ln.bias = nn.Parameter(bias)
    return ln

torch.nn.LayerNorm.load = load_layer_norm


class FastLinear(nn.Module):
    def __init__(
        self,
        weight, bias,
        ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @staticmethod
    def load(config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight") 
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias") 
        else:
            bias = None
        return FastLinear(weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)
        # TODO Is this faster ?
        # prefix_shape = input.shape[:-1]
        # input = input.view(-1, input.shape[-1])
        # if self.bias is not None:
        #     out = torch.addmm(self.bias, input, self.weight)
        # else:
        #     out = torch.matmul(input, self.weight)
        # out = out.view(*prefix_shape, -1)
        # return out


class Linear8bitLt(nn.Module):
    def __init__(self, weight, bias, has_fp16_weights=True,
                       memory_efficient_backward=False, threshold=0.0, index=None):
        super().__init__()
        assert not memory_efficient_backward, "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        self.state = bnb.MatmulLtState()
        self.index = index

        # Necessary for stacked layers
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)
        self.weight.cuda(weight.device)
        self.bias = bias

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out


def get_linear(weight, bias, quantize):
    if quantize is None:
        linear = FastLinear(weight, bias)
    elif quantize == "bitsandbytes":
        linear = Linear8bitLt(
            weight, bias,
            has_fp16_weights=False,
            threshold=6.0,
        )
        if bias is not None:
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
    def load(config, prefix: str, weights):
        weight = weights.get_sharded(f"{prefix}.weight", dim=0) 
        model = TensorParallelHead(get_linear(weight, bias=None, quantize=config.quantize))
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
        if bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias") 
        else:
            bias = None
        layer =  TensorParallelRowLinear(get_linear(weight, bias, config.quantize))
        layer.process_group = weights.process_group
        layer.prefix = prefix
        return layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        torch.distributed.all_reduce(out, group=self.process_group)
        if self.process_group.rank() == 0 and ".0."  in self.prefix:
            logger.info(f"out {self.prefix} {out.view(-1)[:5]}")
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
        if self.process_group.rank() == 0:
            logger.info(f"out {out.view(-1)[:5]}")
        return out

class Embedding(nn.Module):
    def __init__(self, prefix: str, weights):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight") 
        self.weight = nn.Parameter(weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input, self.weight)


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
