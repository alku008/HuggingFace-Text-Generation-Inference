import torch
import torch.distributed

from accelerate import init_empty_weights
from opentelemetry import trace
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig
from typing import Optional, List

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_neox_modeling import (
    FlashGPTNeoXForCausalLM,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights
)

tracer = trace.get_tracer(__name__)


class FlashNeoXSharded(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16
        else:
            raise NotImplementedError("FlashNeoX is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left", truncation_side="left"
        )

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
        )
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device=device, dtype=dtype, process_group=self.process_group)

        model = FlashGPTNeoXForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashCausalLM, self).__init__(
            model=model.to(device),
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
