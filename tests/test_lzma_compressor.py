import lzma
from types import ModuleType

import pytest

from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.compressors.lzma_compressor import LzmaCompressor
from npc_gzip.exceptions import InvalidCompressorException


class TestBz2Compressor:
    base_compressor = BaseCompressor(lzma)
    compressor = LzmaCompressor()
    example_input = "hello there!"

    def test__compress(self) -> None:
        compressed_bytes = self.compressor._compress(self.example_input)
        base_compressed_bytes = self.base_compressor._compress(self.example_input)
        assert compressed_bytes == base_compressed_bytes


    def test_get_compressed_length(self) -> None:
        example_input_length = self.compressor.get_compressed_length(self.example_input)
        assert isinstance(example_input_length, int)
        assert example_input_length > 0


    def test_get_bits_per_character(self) -> None:
        example_bits_per_character = self.compressor.get_bits_per_character(
            self.example_input
        )
        assert isinstance(example_bits_per_character, float)
