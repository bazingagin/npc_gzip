# Compressor Framework
import io
import sys
from importlib import import_module

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


class DefaultCompressor:
    """For non-neural-based compressor"""
    def __init__(self, compressor, typ='text'):
        try:
            self.compressor = import_module(compressor)
        except ModuleNotFoundError:
            raise RuntimeError("Unsupported compressor")
        self.type = typ

    def get_compressed_len(self, x: str) -> int:
        """
        Calculates the size of `x` once compressed.

        Arguments:
            x (str): String to be compressed.

        Returns:
            int: Length of x after compression.
        """
        if self.type == "text":
            return len(self.compressor.compress(x.encode("utf-8")))
        else:
            return len(self.compressor.compress(np.array(x).tobytes()))

    def get_bits_per_char(self, original_fn: str) -> float:
        """
        Returns the compressed size of the original function
        in bits.

        Arguments:
            original_fn (str): Function name to be compressed.

        Returns:
            int: Compressed size of original_fn content in bits.
        """
        with open(original_fn) as fo:
            data = fo.read()
            compressed_str = self.compressor.compress(data.encode("utf-8"))
            return len(compressed_str) * 8 / len(data)


"""Test Compressors"""
if __name__ == '__main__':
    comp = DefaultCompressor('gzip')
    print(comp.get_compressed_len('Hello world'))
