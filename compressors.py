# Compressor Framework
import bz2
import gzip
import lzma


class DefaultCompressor:
    """for non-neural-based compressor"""

    def __init__(self, compressor: str, typ: str = "text"):
        if compressor == "gzip":
            self.compressor = gzip
        elif compressor == "bz2":
            self.compressor = bz2
        elif compressor == "lzma":
            self.compressor = lzma
        else:
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
