from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.exceptions import MissingDependencyException


class Bz2Compressor(BaseCompressor):
    def __init__(self):
        super().__init__(self)

        try:
            import bz2
        except ModuleNotFoundError as e:
            import platform

            major, minor, patch = platform.python_version_tuple()
            if int(major) >= 3 and int(minor) >= 11:
                raise ExceptionGroup([MissingDependencyException("bz2"), e])
            else:
                raise MissingDependencyException("bz2")

        self.compressor = bz2


if __name__ == "__main__":

    compressor = Bz2Compressor()
    example: str = "Hello there!"
    compressed_length: int = compressor.get_compressed_length(example)
    bits_per_character: float = compressor.get_bits_per_character(example)
