from npc_gzip.compressors.base import BaseCompressor

class GZipCompressor(BaseCompressor):
    def __init__(self):
        super().__init__(self)


        try:
            import gzip
        except ModuleNotFoundError as e:
            import platform
            major, minor, patch = platform.python_version_tuple()
            if int(major) >= 3 and int(minor) >= 11:
                raise ExceptionGroup([
                    MissingDependencyException('gzip'),
                    e
                ])
            else:
                raise MissingDependencyException('gzip')

        self.compressor = gzip
        
        
if __name__ == '__main__':

    compressor = GZipCompressor()
    example: str = 'Hello there!'
    compressed_length: int = compressor.get_compressed_length(example)
    bits_per_character: float = compressor.get_bits_per_character(example)
