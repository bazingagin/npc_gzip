class InvalidCompressorException(Exception):
    """
    Is raised when a user is trying to use a compression
    library that is not supported.
    """
    def __init__(self, compression_library: str):
        self.message = f'''
        Compression Library ({compression_library}) 
        is not currently supported.
        '''
        super().__init__(self.message)

class MissingDependencyException(Exception):
    """
    Is raised when an underlying dependency is not
    found when loading a library.
    """
    def __init__(self, compression_library: str):
        self.message = f'''
        Compression Library ({compression_library}) 
        is missing an underlying dependency. Try 
        installing those missing dependencies and 
        load this again. 

        Common missing dependencies for:

        * lzma:
            * brew install xz
            * sudo apt-get install lzma liblzma-dev libbz2-dev

        * bz2:
            * sudo apt-get install lzma liblzma-dev libbz2-dev

        '''
        super().__init__(self.message)