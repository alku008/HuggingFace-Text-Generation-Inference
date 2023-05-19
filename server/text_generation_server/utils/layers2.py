import torch

from torch import nn
from torch.nn import functional as F
from typing import Optional

HAS_BITS_AND_BYTES = True
try:
    from bitsandbytes.nn import Linear8bitLt
except ImportError as e:
    HAS_BITS_AND_BYTES = False


class FastLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(FastLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.quantized = False
        self.bnb_linear = None

    def prepare_weights(self, quantize: Optional[str] = None):
        if quantize == "bitsandbytes":
            if not HAS_BITS_AND_BYTES:
                raise ImportError(
                    "bitsandbytes is not available on your machine either because it is not installed "
                    "or you don't have a GPU.\n"
                    "You can install it with `pip install bitsandbytes`."
                )

            self.quantized = True
            self.bnb_linear = Linear8bitLt(
                self.in_features,
                self.out_features,
                has_fp16_weights=False,
                threshold=6.0,
                bias=False,
            )
            # Copy data to bnb_linear
            self.bnb_linear.weight.data = self.weight.data
            if self.bias is not None:
                self.bnb_linear.bias = nn.Parameter(self.bias)

            # Delete reference to data
            self.weight = None
            self.bias = None
        elif quantize == "gptq":
            raise NotImplementedError("`gptq` is not implemented for now")
        elif quantize is None:
            self.weight = nn.Parameter(self.weight.T)
        else:
            raise ValueError(f"Unexpected quantize `{quantize}`")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.addmm(self.bias, input, self.weight)
        return torch.matmul(input, self.weight)


class TensorParallelRowLinear(FastLinear):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        reduce=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_world_size = process_group.size()
        self.reduce = reduce
        assert in_features % self.tp_world_size == 0
        in_features = in_features // self.tp_world_size

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super(TensorParallelRowLinear, self).forward(input)
        if self.reduce:
            torch.distributed.all_reduce(out, group=self.process_group)

        return out
