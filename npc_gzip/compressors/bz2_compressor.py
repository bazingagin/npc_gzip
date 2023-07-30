from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.exceptions import MissingDependencyException


class Bz2Compressor(BaseCompressor):
    """
    bz2 compressor that inherits from
    `npc_gzip.compressors.base.BaseCompressor`

    >>> compressor: BaseCompressor = Bz2Compressor()
    >>> example: str = "Hello there!"
    >>> compressed_length: int = compressor.get_compressed_length(example)
    >>> bits_per_character: float = compressor.get_bits_per_character(example)
    >>> assert isinstance(compressed_length, int)
    >>> assert isinstance(bits_per_character, float)
    """

    def __init__(self) -> None:
        super().__init__(self)

        try:
            import bz2
        except ModuleNotFoundError as e:
            raise MissingDependencyException("bz2") from e

        self.compressor = bz2
