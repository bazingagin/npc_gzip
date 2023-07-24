# Compressor Framework
import sys
from importlib import import_module
from tqdm import tqdm
import torch.nn.functional as F
import io


class DefaultCompressor:
    """for non-neural-based compressor"""
    def __init__(self, compressor, typ='text'):
        try:
            self.compressor = import_module(compressor)
        except ModuleNotFoundError:
            raise RuntimeError("Unsupported compressor")
        self.type = typ

    def get_compressed_len(self, x: str) -> int:
        """
        Calculates the size of `x` once compressed.

        """
        if self.type == "text":
            return len(self.compressor.compress(x.encode("utf-8")))
        else:
            return len(self.compressor.compress(np.array(x).tobytes()))

    def get_bits_per_char(self, original_fn: str) -> float:
        """
        Returns the compressed size of the original function
        in bits.

        """
        with open(original_fn) as fo:
            data = fo.read()
            compressed_str = self.compressor.compress(data.encode("utf-8"))
            return len(compressed_str) * 8 / len(data)


"""Test Compressors"""
if __name__ == '__main__':
    comp = DefaultCompressor('gzip')
    print(comp.get_compressed_len('Hello world'))
