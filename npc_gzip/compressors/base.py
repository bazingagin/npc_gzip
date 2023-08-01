import os
from types import ModuleType

from npc_gzip.exceptions import InvalidCompressorException


class BaseCompressor:
    """
    Default compressor class that other compressors inherit
    from.

    >>> import gzip

    >>> compressor = BaseCompressor(compressor=gzip)

    >>> example = "Hello there!"
    >>> compressed_length: int = compressor.get_compressed_length(example)
    >>> bits_per_character: float = compressor.get_bits_per_character(example)
    >>> assert isinstance(compressed_length, int)
    >>> assert isinstance(bits_per_character, float)
    """

    def __init__(self, compressor: ModuleType) -> None:
        if not isinstance(compressor, (ModuleType, BaseCompressor)):
            raise InvalidCompressorException(
                f"Not a function passed: {type(compressor)}"
            )
        self.compressor = compressor

    def _open_file(self, filepath: str, as_bytes: bool = False) -> str:
        """
        Helper function that loads and returns the contents
        of a file at `filepath`. Optional `as_bytes` parameter
        will load the file contents with `open(filepath, 'rb')`
        if True.

        Arguments:
            filepath (str): Path to the file to read contents from.
            as_bytes (bool): [Optional] If true, opens the file as bytes.

        Returns:
            str: File contents as a string.
        """

        assert os.path.isfile(filepath), f"Filepath ({filepath}) does not exist."
        file_contents = None
        open_mode = "r"
        if as_bytes:
            open_mode = "rb"

        with open(filepath, open_mode) as f:
            file_contents = f.read()

        if as_bytes:
            file_contents = file_contents.decode("utf-8")

        return file_contents

    def _compress(self, x: str) -> bytes:
        """
        Applies the compression algorithm to `x` and
        returns the results as bytes.

        Arguments:
            x (str): The string you want to compress.

        Returns:
            bytes: The compressed bytes representation of `x`.
        """

        x: bytes = x.encode("utf-8")
        compressed: bytes = self.compressor.compress(x)
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

    def get_bits_per_character(self, x: str) -> float:
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

        compressed_length_per_number_of_characters: float = (
            compressed_length_in_bits / number_of_characters
        )
        return compressed_length_per_number_of_characters
