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
    download_weights,
    weight_hub_files,
    LocalEntryNotFoundError,
)

tracer = trace.get_tracer(__name__)

class Weights:
    def __init__(self, filenames: List[Path], device, dtype, process_group):
        routing = {}
        for filename in filenames:
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing:
                        raise RuntimeError(
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                        )
                    routing[k] = filename
        self.routing = routing
        self.device = device
        self.dtype = dtype
        self.process_group = process_group


    def get_filename(self, tensor_name: str) -> str:
        filename = self.routing.get(tensor_name, None)
        if filename is None:
            raise RuntimeError(f"weight {tensor_name} does not exist")
        return filename

    def get_shape(self, tensor_name: str):
        filename = self.get_filename(tensor_name)
        with safe_open(filename, framework="pytorch") as f:
            slice_ = f.get_slice(tensor_name)
            return slice_.get_shape()

    def get_tensor(self, tensor_name: str):
        filename = self.get_filename(tensor_name)
        with safe_open(filename, framework="pytorch") as f:
            tensor = f.get_tensor(tensor_name)
        tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_sharded(self, tensor_name: str, dim: int):
        filename = self.get_filename(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        with safe_open(filename, framework="pytorch") as f:
            slice_ = f.get_slice(tensor_name)
            size = slice_.get_shape()[dim]
            block_size = size // world_size
            start = rank * block_size
            stop = (rank + 1) * block_size

            if dim == 0:
                tensor = slice_[start:stop]
            elif dim == 1:
                tensor = slice_[:, start:stop]
            else:
                raise NotImplementedError("Let's make that generic when needed")
        tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor


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
