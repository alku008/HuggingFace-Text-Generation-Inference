import torch
import torch.distributed

from accelerate import init_empty_weights
from opentelemetry import trace
from pathlib import Path
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.llama import LlamaTokenizer
from typing import Optional, List

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
    download_weights,
    weight_hub_files,
    LocalEntryNotFoundError,
)

tracer = trace.get_tracer(__name__)

class FlashLlama(FlashCausalLM):
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
            raise NotImplementedError("FlashLlama is only available on GPU")

        tokenizer = LlamaTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
        )

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
        )

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)

        config.quantize = quantize
        model = FlashLlamaForCausalLM(config, weights, process_group=self.process_group)

        torch.distributed.barrier(group=self.process_group)
        super(FlashCausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
