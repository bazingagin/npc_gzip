from typing import Sequence

import numpy as np

from npc_gzip.exceptions import CompressedValuesEqualZero, InvalidShapeException


class Distance:
    """
    Used to calculate the distance between compressed
    objects. Typical usage is your training data as
    `compressed_values_a`, the data you want to predict
    on as `compressed_values_b` and the two values
    concatenated then compressed as `compressed_values_ab`.

    >>> import random

    >>> a = random.random()
    >>> b = random.random()
    >>> ab = random.random()

    >>> distance = Distance(a, b, ab)
    >>> ncd: float = distance._ncd(a, b, ab)
    >>> cdm: float = distance._cdm(a, b, ab)
    >>> clm: float = distance._clm(a, b, ab)
    >>> mse: float = distance._mse(a, b)

    >>> assert isinstance(ncd, float)
    >>> assert isinstance(cdm, float)
    >>> assert isinstance(clm, float)
    >>> assert isinstance(mse, float)

    >>> a = np.random.rand(3, 10)
    >>> b = np.random.rand(3, 10)
    >>> ab = np.random.rand(3, 10)

    >>> distance = Distance(a, b, ab)

    >>> ncd: np.ndarray = distance.ncd
    >>> cdm: np.ndarray = distance.cdm
    >>> clm: np.ndarray = distance.clm
    >>> mse: np.ndarray = distance.mse

    >>> assert isinstance(ncd, np.ndarray)
    >>> assert isinstance(cdm, np.ndarray)
    >>> assert isinstance(clm, np.ndarray)
    >>> assert isinstance(mse, np.ndarray)
    """

    def __init__(
        self,
        compressed_values_a: Sequence,
        compressed_values_b: Sequence,
        compressed_values_ab: Sequence,
    ) -> None:
        if not isinstance(compressed_values_a, np.ndarray):
            compressed_values_a = np.array(compressed_values_a)

        if not isinstance(compressed_values_b, np.ndarray):
            compressed_values_b = np.array(compressed_values_b)

        if not isinstance(compressed_values_ab, np.ndarray):
            compressed_values_ab = np.array(compressed_values_ab)

        if (
            compressed_values_a.shape
            == compressed_values_b.shape
            == compressed_values_ab.shape
        ):
            self.compressed_values_a = compressed_values_a
            self.compressed_values_b = compressed_values_b
            self.compressed_values_ab = compressed_values_ab
        else:
            raise InvalidShapeException(
                compressed_values_a,
                compressed_values_b,
                compressed_values_ab,
                function_name="Distance.__init__",
            )

    def _ncd(
        self,
        compressed_value_a: float,
        compressed_value_b: float,
        compressed_value_ab: float,
    ) -> float:
        denominator = max(compressed_value_a, compressed_value_b)
        if denominator == 0:
            raise CompressedValuesEqualZero(
                compressed_value_a, compressed_value_b, function_name="Distance._ncd"
            )

        numerator = compressed_value_ab - min(compressed_value_a, compressed_value_b)
        distance = numerator / denominator
        return distance

    def _cdm(
        self,
        compressed_value_a: float,
        compressed_value_b: float,
        compressed_value_ab: float,
    ) -> float:
        denominator = compressed_value_a + compressed_value_b
        if denominator == 0:
            raise CompressedValuesEqualZero(
                compressed_value_a, compressed_value_b, function_name="Distance._cdm"
            )

        numerator = compressed_value_ab

        distance = numerator / denominator
        return distance

    def _clm(
        self,
        compressed_value_a: float,
        compressed_value_b: float,
        compressed_value_ab: float,
    ) -> float:
        denominator = compressed_value_ab
        if denominator == 0:
            raise CompressedValuesEqualZero(
                compressed_value_ab, function_name="Distance._clm"
            )

        numerator = 1 - (compressed_value_a + compressed_value_b - compressed_value_ab)

        distance = numerator / denominator
        return distance

    def _mse(
        self,
        compressed_value_a: Sequence,
        compressed_value_b: Sequence,
    ) -> np.ndarray:
        """
        Computes the mean squared error between two
        values.

        """

        if not isinstance(compressed_value_a, np.ndarray):
            compressed_value_a = np.array(compressed_value_a)

        if not isinstance(compressed_value_b, np.ndarray):
            compressed_value_b = np.array(compressed_value_b)

        compressed_value_a = compressed_value_a.reshape(-1)
        compressed_value_b = compressed_value_b.reshape(-1)

        numerator = np.sum((compressed_value_a - compressed_value_b) ** 2)
        denominator = compressed_value_a.shape[0]

        if denominator == 0:
            raise CompressedValuesEqualZero(
                compressed_value_a, compressed_value_b, function_name="Distance._mse"
            )

        mse = numerator / denominator
        return mse

    @property
    def ncd(self) -> np.ndarray:
        """
        A numpy vectorized form of self._ncd.

        """

        out = np.vectorize(self._ncd)(
            self.compressed_values_a,
            self.compressed_values_b,
            self.compressed_values_ab,
        )
        out = out.reshape(self.compressed_values_a.shape)

        return out

    @property
    def cdm(self) -> np.ndarray:
        """
        A numpy vectorized form of self._cdm.

        """

        out = np.vectorize(self._cdm)(
            self.compressed_values_a,
            self.compressed_values_b,
            self.compressed_values_ab,
        )
        out = out.reshape(self.compressed_values_a.shape)

        return out

    @property
    def clm(self) -> np.ndarray:
        """
        A numpy vectorized form of self._clm.

        """

        out = np.vectorize(self._clm)(
            self.compressed_values_a,
            self.compressed_values_b,
            self.compressed_values_ab,
        )
        out = out.reshape(self.compressed_values_a.shape)

        return out

    @property
    def mse(self) -> np.ndarray:
        """
        A numpy vectorized form of self._mse.

        """

        out = np.vectorize(self._mse)(
            self.compressed_values_a,
            self.compressed_values_b,
        )
        out = out.reshape(self.compressed_values_a.shape)

        return out
