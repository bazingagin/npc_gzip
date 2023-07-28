from typing import Sequence, Union

import numpy as np

from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.distance import Distance
from npc_gzip.exceptions import (
    InputLabelEqualLengthException,
    InvalidObjectTypeException,
    UnsupportedDistanceMetricException,
)


class KnnCompressor:
    """
    Given the training input and optional
    training labels data, this class stores
    the data in memory

    """

    def __init__(
        self,
        compressor: BaseCompressor,
        training_inputs: Union[np.ndarray, Sequence[str]],
        training_labels: Union[np.ndarray, Sequence[Union[str, int]]] = None,
        distance_metric: str = "ncd",
    ):
        self.compressor = compressor

        if isinstance(training_inputs, list) and isinstance(training_labels, list):
            if training_labels is not None:
                if len(training_inputs) != len(training_labels):
                    raise InputLabelEqualLengthException(
                        len(training_inputs),
                        len(training_labels),
                        function_name="KnnGzip.__init__",
                    )

            self.training_inputs: np.ndarray = np.array(training_inputs).reshape(-1)
            self.training_labels: np.ndarray = np.array(training_labels).reshape(-1)

        elif isinstance(training_inputs, np.ndarray) or isinstance(
            training_labels, np.ndarray
        ):
            self.training_inputs: np.ndarray = (
                np.array(training_inputs)
                if isinstance(training_inputs, list)
                else training_inputs
            )
            self.training_labels: np.ndarray = (
                np.array(training_labels)
                if isinstance(training_labels, list)
                else training_labels
            )

            self.training_inputs = self.training_inputs.reshape(-1)
            self.training_labels = self.training_labels.reshape(-1)

        else:
            raise InvalidObjectTypeException(
                type(training_inputs),
                supported_types=[list, np.ndarray],
                function_name="KnnGzip.__init__",
            )

        assert (
            self.training_inputs.shape == self.training_labels.shape
        ), f"""
        Training Inputs and Labels did not maintain their 
        shape during the conversion from lists to numpy arrays.
        This is most likely a bug in the numpy package:
        
        self.training_inputs.shape: {self.training_inputs.shape}
        self.training_labels.shape: {self.training_labels.shape}
        """

        self.supported_distance_metrics: list = ["ncd", "clm", "cdm", "mse"]

        if distance_metric not in self.supported_distance_metrics:
            raise UnsupportedDistanceMetricException(
                distance_metric,
                self.supported_distance_metrics,
                function_name="KnnGzip.__init__",
            )

        self.distance_metric = distance_metric

        self.compressed_training_inputs: list = [
            self.compressor.get_compressed_length(data) for data in self.training_inputs
        ]
        self.compressed_training_inputs: np.ndarray = np.array(
            self.compressed_training_inputs
        ).reshape(-1)

    def _calculate_distance(
        self, compressed_x: np.ndarray, compressed_combined: np.ndarray
    ) -> np.ndarray:
        """
        Helper function that converts the string representation
        of `self.distance_metric` to the actual
        `npc_gzip.distance.Distance.[distance_metric]`. Then
        that distance metric is calculated using
        `self.compressed_training_inputs`, `compressed_x` and
        `compressed_combined`.

        Arguments:
            compressed_x (np.ndarray): Numpy array representing the
                                       compressed lengths of the input
                                       data to be scored.
            compressed_combined (np.ndarray): Numpy array representing the
                                              compressed lengths of the input
                                              data combined with
                                              each training sample to be scored.
        Returns:
            np.ndarray: Numpy array containing the distance metric.
        """

        distance = Distance(
            np.resize(self.compressed_training_inputs, compressed_x.shape),
            compressed_x,
            compressed_combined,
        )

        if self.distance_metric == "ncd":
            return distance.ncd
        elif self.distance_metric == "clm":
            return distance.clm
        elif self.distance_metric == "cdm":
            return distance.cdm
        elif self.distance_metric == "mse":
            return distance.mse
        else:
            return "Invalid Distance Metric"

    def _compress_sample(self, x: str) -> tuple:
        """
        Helper method that compresses `x` against each
        item in `self.training_inputs` and returns the
        distance between each sample using the
        self.distance_metric from the
        `npc_gzip.distance.Distance` object.

        Arguments:
            x (np.ndarray): The sample data to compare against
                            the `training_inputs`.

        Returns:
            np.ndarray: Compressed length of `x` as an array of shape
                        [self.compressed_training_inputs.shape[0]].
            np.ndarray: Compressed length of the combination of `x` and
                        each training sample as an array of shape
                        [self.compressed_training_inputs.shape[0]].
        """

        assert isinstance(
            x, str
        ), f"Non-string was passed to self._compress_sample: {x}"
        x_compressed = self.compressor.get_compressed_length(x)
        compressed_x: list = [
            x_compressed for _ in range(self.training_inputs.shape[0])
        ]
        compressed_x: np.ndarray = np.array(compressed_x).reshape(-1)
        assert compressed_x.shape == self.training_inputs.shape

        combined: list = []
        for training_sample in self.training_inputs:
            train_and_x: str = self.concatenate_with_space(training_sample, x)
            combined_compressed: int = self.compressor.get_compressed_length(
                train_and_x
            )
            combined.append(combined_compressed)

        combined: np.ndarray = np.array(combined).reshape(-1)
        assert self.training_inputs.shape == compressed_x.shape == combined.shape

        return (compressed_x, combined)

    @staticmethod
    def concatenate_with_space(stringa: str, stringb: str) -> str:
        """
        Combines `stringa` and `stringb` with a space.

        Arguments:
            stringa (str): First item.
            stringb (str): Second item.

        Returns:
            str: `{stringa} {stringb}`
        """

        stringa = str(stringa)
        stringb = str(stringb)

        return stringa + " " + stringb

    def predict(self, x: Union[np.ndarray, Sequence[str], str], top_k: int = 1):
        """
        Given a test sample `x`, this method will
        compare `x` against the training data provided
        and will return the best matching label if
        `training_labels` was passed during instantiation.
        If `training_labels` was not passed during
        instantiation, or if `top_k` is not None, the most
        similar samples from `training_inputs` will be
        returned.

        resulting array is [batch_size, self.training_inputs.shape[0] ]

        Arguments:
            x (Union[np.ndarray, Sequence[str]]): The sample data to compare against
                                                  the `training_inputs`.
            top_k (int): [Optional] If not None, the most similar
                         `k` number of samples will be returned.
                         `top_k` must be greater than zero.
                         [default: top_k=1]

        Returns:
            np.ndarray: The distance-metrics matrix computed on the test 
                        set.
            np.ndarray: The `top_k` most similar `self.training_labels`.
            np.ndarray: The `top_k` best matching samples
                           from `self.training_inputs`.
        """

        if not isinstance(x, np.ndarray):
            x: np.ndarray = np.array(x)

        assert top_k > 0, f"top_k ({top_k}) must be greater than zero."

        x = x.reshape(-1)
        assert (
            top_k <= x.shape[0]
        ), f"""
        top_k ({top_k}) must be less or equal to than the number of 
        samples provided to be predicted on ({x.shape[0]})
        
        """

        compressed_samples = []
        compressed_combined = []
        for sample in x:
            compressed_sample, combined = self._compress_sample(sample)
            compressed_samples.append(compressed_sample)
            compressed_combined.append(combined)

        compressed_samples: np.ndarray = np.array(compressed_samples)
        compressed_combined: np.ndarray = np.array(compressed_combined)

        distances: np.ndarray = self._calculate_distance(
            compressed_samples, compressed_combined
        )

        # top matching training samples and labels by
        # minimum distance.

        # get indicies of minimum top_k distances.
        minimum_distance_indices = np.argpartition(distances, top_k)[:, :top_k]

        similar_samples = self.training_inputs[minimum_distance_indices]
        labels = []
        if self.training_labels is not None:
            labels = self.training_labels[minimum_distance_indices]

        return distances, labels, similar_samples


if __name__ == "__main__":
    import random

    from npc_gzip.compressors.gzip_compressor import GZipCompressor

    training_data = ["hey", "hi", "how are you?", "not too bad"]

    training_labels = [random.randint(0, 1) for _ in range(len(training_data))]
    assert len(training_data) == len(training_labels)

    model = KnnCompressor(
        compressor=GZipCompressor(),
        training_inputs=training_data,
        training_labels=training_labels,
        distance_metric="ncd",
    )

    test = np.array(["hey", "you are a real pain in my ass", "go away please"])

    top_k = 1
    distances, labels, similar_samples = model.predict(test, top_k=top_k)
    assert distances.shape == (test.shape[0], len(training_data))
    assert labels.shape == similar_samples.shape
    assert distances.shape[0] == labels.shape[0] == similar_samples.shape[0]
