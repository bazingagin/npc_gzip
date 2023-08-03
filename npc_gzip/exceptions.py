from typing import Any, Optional

import numpy as np


class InvalidCompressorException(Exception):
    """
    Is raised when a user is trying to use a compression
    library that is not supported.
    """

    def __init__(self, compression_library: str) -> None:
        self.message = f"""
        Compression Library ({compression_library})
        is not currently supported.
        """
        super().__init__(self.message)


class MissingDependencyException(Exception):
    """
    Is raised when an underlying dependency is not
    found when loading a library.
    """

    def __init__(self, compression_library: str) -> None:
        self.message = f"""
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

        """
        super().__init__(self.message)


class StringTooShortException(Exception):
    def __init__(
        self, stringa: str, stringb: str, function_name: Optional[str] = None
    ) -> None:
        self.message = f"""
        Unable to aggregate ({stringa}) and ({stringb}).
        One or both of the two strings are too short to concatenate.

        """

        if function_name is not None:
            self.message += f"function_name: {function_name}"

        super().__init__(self.message)


class CompressedValuesEqualZero(Exception):
    def __init__(
        self,
        compressed_value_a: float,
        compressed_value_b: Optional[float] = None,
        function_name: Optional[str] = None,
    ) -> None:
        self.message = """
        The combination of compressed values passed equal zero.
        This will result in a divide by zero error.


        """

        if function_name is not None:
            self.message += f"function_name: {function_name}"
        super().__init__(self.message)


class AllOrNoneException(Exception):
    def __init__(
        self,
        a: Any,
        b: Any,
        c: Any,
        function_name: Optional[str] = None,
    ) -> None:
        self.message = f"""
        The passed values must either all be None or not None.
            arg1: {type(a)}
            arg2: {type(b)}
            arg3: {type(c)}

        """

        if function_name is not None:
            self.message += f"function_name: {function_name}"
        super().__init__(self.message)


class InvalidShapeException(Exception):
    def __init__(
        self,
        array_a: np.ndarray,
        array_b: np.ndarray,
        array_c: np.ndarray,
        function_name: Optional[str] = None,
    ) -> None:
        self.message = f"""
        The passed values must either all of the same shape.
            arg1: {array_a.shape}
            arg2: {array_b.shape}
            arg3: {array_c.shape}

        """

        if function_name is not None:
            self.message += f"function_name: {function_name}"
        super().__init__(self.message)


class UnsupportedDistanceMetricException(Exception):
    def __init__(
        self,
        distance_metric: str,
        supported_distance_metrics: Optional[list] = None,
        function_name: Optional[str] = None,
    ) -> None:
        self.message = f"""
        The `distance_metric` ({distance_metric}) provided is not
        currently supported. Please submit an Issue and/or
        Pull Request here to add support:
        https://github.com/bazingagin/npc_gzip

        """

        if supported_distance_metrics is not None:
            self.message += (
                f"supported_distance_metrics: {supported_distance_metrics}\n"
            )

        if function_name is not None:
            self.message += f"function_name: {function_name}"
        super().__init__(self.message)


class InvalidObjectTypeException(TypeError):
    def __init__(
        self,
        passed_type: str,
        supported_types: Optional[list] = None,
        function_name: Optional[str] = None,
    ) -> None:
        self.message = f"""
        The type passed ({passed_type}) provided is not
        currently supported.

        """

        if supported_types is not None:
            self.message += f"supported types: {supported_types}\n"

        if function_name is not None:
            self.message += f"function_name: {function_name}"
        super().__init__(self.message)


class InputLabelEqualLengthException(Exception):
    def __init__(
        self,
        training_samples: int,
        label_samples: int,
        function_name: Optional[str] = None,
    ) -> None:
        self.message = f"""
        If training labels are passed, the number
        of training data samples must equal the
        number of training label samples

        training_samples: {training_samples}
        label_samples: {label_samples}

        """

        if function_name is not None:
            self.message += f"function_name: {function_name}"
        super().__init__(self.message)
