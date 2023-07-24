import gzip
from npc_gzip.compressors.base import BaseCompressor

class GZipCompressor(BaseCompressor):
    def __init__(self):
        super().__init__(self)

        self.compressor = gzip
        
        
if __name__ == '__main__':

    compressor = GZipCompressor()
    example: str = 'Hello there!'
    compressed_length: int = compressor.get_compressed_length(example)
    bits_per_character: float = compressor.get_bits_per_char(example)
