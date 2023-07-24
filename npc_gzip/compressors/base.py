from types import ModuleType
from npc_gzip.exceptions import InvalidCompressorException

class BaseCompressor:
    """
    Default compressor class that other compressors inherit
    from.

    """

    def __init__(self, compressor: ModuleType):
        self.compressor = compressor

    def _compress(self, x: str) -> bytes:
        """
        Applies the compression algorithm to `x` and
        returns the results as bytes.

        Arguments:
            x (str): The string you want to compress.

        Returns:
            bytes: The compressed bytes representation of `x`.
        """

        x = str(x)
        x = x.encode('utf-8')
        compressed = self.compressor.compress(x)
        return compressed

    def get_compressed_length(self, x: str) -> int:
        """
        Calculates the size of `x` once compressed.

        Arguments:
            x (str): String you want to compute the compressed length of.

        Returns:
            int: Length of `x` once compressed.
        """

        compressed_value: bytes = self._compress(x)
        compressed_length: int = len(compressed_value)
        return compressed_length

    def get_bits_per_char(self, x: str) -> float:
        """
        Returns the compressed size of `x` relative to 
        the number of characters in string `x`.

        Arguments:
            x (str): String you want to compute the compressed length per
                     character in.

        Returns:
            float: Length of `x` once compressed divided by the number of 
                 characters in `x`.
        """

        compressed_value: bytes = self._compress(x)
        compressed_length: int = len(compressed_value)
        compressed_length_in_bits: int = compressed_length * 8
        number_of_characters: int = len(x)

        compressed_length_per_number_of_characters: float = compressed_length_in_bits / number_of_characters
        return compressed_length_per_number_of_characters


if __name__ == "__main__":

    import gzip
    compressor = BaseCompressor(compressor=gzip)

    example = 'Hello there!'
    compressed_length: int = compressor.get_compressed_length(example)
    bits_per_character: float = compressor.get_bits_per_char(example)

    print(compressed_length, bits_per_character)