from typing import Optional, Sequence

import numpy as np
from tqdm import tqdm

from npc_gzip.aggregations import concatenate_with_space
from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.distance import Distance
from npc_gzip.exceptions import (
    InputLabelEqualLengthException,
    InvalidObjectTypeException,
    UnsupportedDistanceMetricException,
)


class KnnClassifier:
    """
    Given the training input and optional
    training labels data, this class stores
    the data in memory

    >>> import random
    >>> from npc_gzip.compressors.gzip_compressor import GZipCompressor

    >>> training_data = ["hey", "hi", "how are you?", "not too bad"]

    >>> training_labels = [random.randint(0, 1) for _ in range(len(training_data))]
    >>> assert len(training_data) == len(training_labels)

    >>> model = KnnClassifier(
    ...     compressor=GZipCompressor(),
    ...     training_inputs=training_data,
    ...     training_labels=training_labels,
    ...     distance_metric="ncd",
    ... )

    >>> test = np.array(["hey", "you are a real pain in my ass", "go away please"])

    >>> top_k = 1
    >>> distances, labels, similar_samples = model.predict(test, top_k=top_k)
    >>> assert distances.shape == (test.shape[0], len(training_data))
    >>> assert labels.shape == (test.shape[0], )
    >>> assert distances.shape[0] == labels.shape[0] == similar_samples.shape[0]
    """

    def __init__(
        self,
        compressor: BaseCompressor,
        training_inputs: Sequence,
        training_labels: Optional[Sequence] = None,
        distance_metric: str = "ncd",
    ) -> None:
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
        self,
        compressed_input: np.ndarray,
        compressed_combined: np.ndarray,
        compressed_training: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Helper function that converts the string representation
        of `self.distance_metric` to the actual
        `npc_gzip.distance.Distance.[distance_metric]`. Then
        that distance metric is calculated using
        `self.compressed_training_inputs`, `compressed_input` and
        `compressed_combined`.

        Arguments:
            compressed_input (np.ndarray): Numpy array representing the
                                       compressed lengths of the input
                                       data to be scored.
            compressed_combined (np.ndarray): Numpy array representing the
                                              compressed lengths of the input
                                              data combined with
                                              each training sample to be scored.
        Returns:
            np.ndarray: Numpy array containing the distance metric.
        """

        if compressed_training is None:
            distance = Distance(
                np.resize(self.compressed_training_inputs, compressed_input.shape),
                compressed_input,
                compressed_combined,
            )
        else:
            distance = Distance(
                np.resize(compressed_training, compressed_input.shape),
                compressed_input,
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
            raise UnsupportedDistanceMetricException(
                self.distance_metric,
                supported_distance_metrics=self.supported_distance_metrics,
                function_name="_calculate_distance",
            )

    def _compress_sample(
        self,
        x: str,
        training_inputs: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Helper method that compresses `x` against each
        item in `self.training_inputs` and returns the
        distance between each sample using the
        self.distance_metric from the
        `npc_gzip.distance.Distance` object.

        Arguments:
            x (np.ndarray): The sample data to compare against
                            the `training_inputs`.
            training_inputs (np.ndarray): [Optional] If provided, this
                                          method will use `training_inputs`
                                          when calculating the distance matrix
                                          rather than `self.training_inputs`.

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

        if training_inputs is None:
            training_inputs = self.training_inputs

        compressed_input_length: int = self.compressor.get_compressed_length(x)
        compressed_input: list = [
            compressed_input_length for _ in range(training_inputs.shape[0])
        ]
        compressed_input: np.ndarray = np.array(compressed_input).reshape(-1)
        assert compressed_input.shape == training_inputs.shape

        combined: list = []
        for training_sample in training_inputs:
            train_and_x: str = concatenate_with_space(training_sample, x)
            combined_compressed: int = self.compressor.get_compressed_length(
                train_and_x
            )
            combined.append(combined_compressed)

        combined: np.ndarray = np.array(combined).reshape(-1)
        assert training_inputs.shape == compressed_input.shape == combined.shape

        return (compressed_input, combined)

    def sample_data(
        self, sampling_percentage: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given a `sampling_percentage`, this method randomly
        samples data from `self.training_inputs` &
        `self.training_labels` (if exists) without replacement
        and returns two numpy arrays containing the randomly
        sampled data.

        Arguments:
            sampling_percentage (float): (0, 1.0] % of data to
                                         randomly sample from the
                                         training inputs and labels.

        Returns:
            np.ndarray: Randomly sampled training inputs.
            np.ndarray: Randomly sampled training labels.
            np.ndarray: Indices that the training inputs &
                        labels were sampled.
        """

        total_inputs: int = self.training_inputs.shape[0]
        sample_size: int = int(sampling_percentage * total_inputs)
        randomly_sampled_indices: np.ndarray = np.random.choice(
            total_inputs, sample_size, replace=False
        )

        randomly_sampled_inputs: np.ndarray = self.training_inputs[
            randomly_sampled_indices
        ]
        randomly_sampled_labels: np.ndarray = np.array([])
        if self.training_labels is not None:
            randomly_sampled_labels = self.training_labels[randomly_sampled_indices]

        return (
            randomly_sampled_inputs,
            randomly_sampled_labels,
            randomly_sampled_indices,
        )

    def predict(
        self,
        x: Sequence,
        top_k: int = 1,
        sampling_percentage: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Faster version of `predict`. This method
        will compare `x` against a sample of the
        training data provided and will return the
        best matching label if `training_labels` was
        passed during instantiation.

        If `training_labels` was not passed during
        instantiation, or if `top_k` is not None, the most
        similar samples from `training_inputs` will be
        returned.

        resulting array is [batch_size, self.training_inputs.shape[0] ]

        Arguments:
            x (Sequence): The sample data to compare against
                                                  the `training_inputs`.
            top_k (int): [Optional] If not None, the most similar
                         `k` number of samples will be returned.
                         `top_k` must be greater than zero.
                         [default: top_k=1]
            sampling_percentage (float): (0.0, 1.0] % of `self.training_inputs`
                                         to sample predictions against.

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

        x: np.ndarray = x.reshape(-1)
        assert (
            top_k <= x.shape[0]
        ), f"""
        top_k ({top_k}) must be less or equal to than the number of
        samples provided to be predicted on ({x.shape[0]})

        """

        # sample training inputs and labels
        training_inputs, training_labels, randomly_sampled_indices = self.sample_data(
            sampling_percentage
        )

        samples: list = []
        combined: list = []
        for sample in tqdm(x, desc="Compressing input..."):
            compressed_sample, combined_length = self._compress_sample(
                sample, training_inputs=training_inputs
            )
            samples.append(compressed_sample)
            combined.append(combined_length)

        compressed_samples: np.ndarray = np.array(samples)
        compressed_combined: np.ndarray = np.array(combined)

        assert isinstance(training_inputs, np.ndarray)
        assert isinstance(compressed_samples, np.ndarray)
        assert isinstance(compressed_combined, np.ndarray)
        assert isinstance(self.compressed_training_inputs, np.ndarray)

        compressed_training: np.ndarray = self.compressed_training_inputs[
            randomly_sampled_indices
        ]

        distances: np.ndarray = self._calculate_distance(
            compressed_samples, compressed_combined, compressed_training
        )

        # top matching training samples and labels by
        # minimum distance.

        # get indicies of minimum top_k distances.
        minimum_distance_indices = np.argpartition(distances, top_k)[:, :top_k]

        similar_samples: np.ndarray = training_inputs[minimum_distance_indices]
        labels: np.ndarray = training_labels[minimum_distance_indices]
        predicted_labels = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=labels
        )

        return distances, predicted_labels, similar_samples
