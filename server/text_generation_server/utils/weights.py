from pathlib import Path
from typing import Optional, List
from safetensors import safe_open

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


