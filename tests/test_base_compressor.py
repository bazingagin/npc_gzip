import gzip
from types import ModuleType

import pytest

from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.exceptions import InvalidCompressorException


class TestBaseCompressor:
    compressor = BaseCompressor(compressor=gzip)
    example_input = "hello there!"

    def test_types(self) -> None:
        invalid_compressors = ["test", 0.1, 123, ("hello", "there"), {"hey": "hi"}]

        for compressor in invalid_compressors:
            with pytest.raises(InvalidCompressorException):
                BaseCompressor(compressor)

        valid_compressor = gzip
        BaseCompressor(valid_compressor)

    def test__open_file(self) -> None:
        valid_filepath = "tests/test_base_compressor.py"
        invalid_filepath = "tests/test_base_compressor.py.xyz"

        with pytest.raises(AssertionError):
            self.compressor._open_file(invalid_filepath)

        with pytest.raises(AssertionError):
            self.compressor._open_file(invalid_filepath, as_bytes=True)

        file_contents = self.compressor._open_file(valid_filepath, as_bytes=False)
        assert isinstance(file_contents, str)

        file_contents = self.compressor._open_file(valid_filepath, as_bytes=True)
        assert isinstance(file_contents, str)

    def test__compress(self) -> None:
        compressed_bytes = self.compressor._compress(self.example_input)
        assert isinstance(compressed_bytes, bytes)


    def test_get_compressed_length(self) -> None:
        example_input_length = self.compressor.get_compressed_length(self.example_input)
        assert isinstance(example_input_length, int)
        assert example_input_length > 0


    def test_get_bits_per_character(self) -> None:
        example_bits_per_character = self.compressor.get_bits_per_character(
            self.example_input
        )
        assert isinstance(example_bits_per_character, float)
