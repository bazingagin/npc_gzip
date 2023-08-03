import random

import numpy as np
import pytest

from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.compressors.bz2_compressor import Bz2Compressor
from npc_gzip.compressors.gzip_compressor import GZipCompressor
from npc_gzip.compressors.lzma_compressor import LzmaCompressor
from npc_gzip.exceptions import (
    InputLabelEqualLengthException,
    UnsupportedDistanceMetricException,
)
from npc_gzip.knn_classifier import KnnClassifier
from npc_gzip.utils import generate_dataset


class TestKnnClassifier:
    gzip_compressor: BaseCompressor = GZipCompressor()
    bz2_compressor: BaseCompressor = Bz2Compressor()
    lzma_compressor: BaseCompressor = LzmaCompressor()

    dataset_size: int = 100
    training_dataset: list = generate_dataset(dataset_size)
    training_labels: list = [random.randint(0, 10) for _ in range(dataset_size)]

    model = KnnClassifier(
        compressor=gzip_compressor,
        training_inputs=training_dataset,
        training_labels=training_labels,
        distance_metric="ncd",
    )

    sample_input: str = "hello there"

    def test_init(self) -> None:
        assert self.model.distance_metric == "ncd"
        assert isinstance(self.model.training_inputs, np.ndarray)
        assert isinstance(self.model.compressed_training_inputs, np.ndarray)

        assert (
            self.model.training_inputs.shape[0]
            == self.model.training_labels.shape[0]
            == self.model.compressed_training_inputs.shape[0]
        )

        assert self.model.supported_distance_metrics == ["ncd", "clm", "cdm", "mse"]

        model = KnnClassifier(
            compressor=self.gzip_compressor,
            training_inputs=np.array(self.training_dataset),
            training_labels=np.array(self.training_labels),
            distance_metric="ncd",
        )

        assert isinstance(model.training_inputs, np.ndarray)
        assert isinstance(model.compressed_training_inputs, np.ndarray)
        assert (
            model.training_inputs.shape[0]
            == model.training_labels.shape[0]
            == model.compressed_training_inputs.shape[0]
        )

    def test_invalid_metric(self) -> None:
        with pytest.raises(UnsupportedDistanceMetricException):
            KnnClassifier(
                compressor=self.gzip_compressor,
                training_inputs=np.array(self.training_dataset),
                training_labels=np.array(self.training_labels),
                distance_metric="hoopla",
            )

    def test_invalid_data_and_label_size(self) -> None:
        training_data_size = 10
        training_label_size = training_data_size - 1

        dataset = generate_dataset(training_data_size)
        labels = [random.randint(0, 10) for _ in range(training_label_size)]

        with pytest.raises(InputLabelEqualLengthException):
            KnnClassifier(
                compressor=self.gzip_compressor,
                training_inputs=dataset,
                training_labels=labels,
                distance_metric="ncd",
            )

    def test__compress_sample(self) -> None:
        compressed_input, compressed_combined = self.model._compress_sample(
            self.sample_input
        )

        assert isinstance(compressed_input, np.ndarray)
        assert isinstance(compressed_combined, np.ndarray)
        assert compressed_input.shape == compressed_combined.shape

    def test__calculate_distance(self) -> None:
        compressed_input, compressed_combined = self.model._compress_sample(
            self.sample_input
        )
        distance = self.model._calculate_distance(compressed_input, compressed_combined)

        assert isinstance(distance, np.ndarray)
        assert distance.shape == compressed_input.shape == compressed_combined.shape

    def test_predict(self) -> None:
        top_k = 1
        (distance, labels, similar_samples) = self.model.predict(
            self.sample_input, top_k
        )

        assert distance.shape == (
            1,
            self.model.training_inputs.shape[0],
        )
        assert labels.shape == (1,)
        assert similar_samples.shape == (top_k, 1)

        test_set_size = random.randint(2, 50)
        test_set = [self.sample_input for _ in range(test_set_size)]
        top_k = 2
        (distance, labels, similar_samples) = self.model.predict(test_set, top_k)

        assert distance.shape == (test_set_size, self.model.training_inputs.shape[0])
        assert labels.shape == (test_set_size,)
        assert similar_samples.shape == (test_set_size, top_k)

    def test_negative_top_k(self) -> None:
        test_set_size = random.randint(1, 50)
        test_set = [self.sample_input for _ in range(test_set_size)]
        top_k = -1

        with pytest.raises(AssertionError):
            self.model.predict(test_set, top_k)

    def test_top_k_bigger_than_test_set(self) -> None:
        test_set_size = random.randint(1, 10)
        test_set = [self.sample_input for _ in range(test_set_size)]
        top_k = test_set_size + 1
        with pytest.raises(AssertionError):
            self.model.predict(test_set, top_k)

    def test_sampling_percentage(self) -> None:
        test_set_size = random.randint(1, 10)
        test_set = [self.sample_input for _ in range(test_set_size)]
        top_k = 1
        (full_distance, full_labels, full_similar_samples) = self.model.predict(
            test_set, top_k, sampling_percentage=1.0
        )
        (distance, labels, similar_samples) = self.model.predict(
            test_set, top_k, sampling_percentage=0.1
        )

        assert full_labels.shape == labels.shape
        assert full_similar_samples.shape == similar_samples.shape

    def test_sample_data(self) -> None:
        sampling_percentage: float = 0.8
        (
            randomly_sampled_inputs,
            randomly_sampled_labels,
            randomly_sampled_indices,
        ) = self.model.sample_data(sampling_percentage)

        assert (
            randomly_sampled_inputs.shape
            == randomly_sampled_labels.shape
            == randomly_sampled_indices.shape
        )
        assert randomly_sampled_inputs.shape[0] == int(
            self.model.training_inputs.shape[0] * sampling_percentage
        )

        assert (
            self.model.training_inputs[randomly_sampled_indices]
            == randomly_sampled_inputs
        ).all()
