from types import ModuleType

import pytest

from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.exceptions import InvalidCompressorException

import gzip
from npc_gzip.compressors.gzip_compressor import GZipCompressor

class TestBz2Compressor:

    base_compressor = BaseCompressor(gzip)
    compressor = GZipCompressor()
    example_input = "hello there!"

    def test__compress(self):
        compressed_bytes = self.compressor._compress(self.example_input)
        base_compressed_bytes = self.base_compressor._compress(self.example_input)
        assert compressed_bytes == base_compressed_bytes

        example_inputs = [0, 0.1, "hey there", [0], (0, 1), {"test": "yep"}]

        for input_ in example_inputs:
            out = self.base_compressor._compress(input_)
            assert isinstance(out, bytes)

    def test_get_compressed_length(self):

        example_input_length = self.compressor.get_compressed_length(self.example_input)
        assert isinstance(example_input_length, int)
        assert example_input_length > 0

        example_inputs = [0, 0.1, "hey there", [0], (0, 1), {"test": "yep"}]

        for input_ in example_inputs:
            out = self.compressor.get_compressed_length(input_)
            assert isinstance(out, int)
            assert out > 0

    def test_get_bits_per_character(self):

        example_bits_per_character = self.compressor.get_bits_per_character(
            self.example_input
        )
        assert isinstance(example_bits_per_character, float)
        example_inputs = [0, 0.1, "hey there", [0], (0, 1), {"test": "yep"}, -1]

        for input_ in example_inputs:
            out = self.compressor.get_bits_per_character(input_)
            assert isinstance(out, float)
            assert out > 0
