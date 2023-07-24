# Experiment framework
import operator
import os
import pickle
import random
import statistics
from collections import Counter, defaultdict
from copy import deepcopy
from functools import partial
from itertools import repeat
from statistics import mode
from typing import Callable

import numpy as np
import torch
from sklearn.metrics.cluster import (adjusted_rand_score,
                                     normalized_mutual_info_score)
from tqdm import tqdm

from compressors import DefaultCompressor


class KnnExpText:
    def __init__(
        self,
        aggregation_function: Callable,
        compressor: DefaultCompressor,
        distance_function: Callable,
    ):
        self.aggregation_func = aggregation_function
        self.compressor = compressor
        self.distance_func = distance_function
        self.distance_matrix = []

    def calc_dis(self, data: list, train_data: list = None, fast: bool = False) -> None:
        """
        Calculates the distance between either `data` and itself or `data` and `train_data`
        and appends the distance to `self.distance_matrix`.

        Arguments:
            data (list): Data to compute distance between.
            train_data (list): [Optional] Training data to compute distance from `data`.
            fast (bool): [Optional] Uses the _fast compression length function of `self.compressor`.

        Returns:
            None: None
        """

        data_to_compare = data
        if train_data is not None:
            data_to_compare = train_data

        for i, t1 in tqdm(enumerate(data)):
            distance4i = []
            if fast:
                t1_compressed = self.compressor.get_compressed_len_fast(t1)
            else:
                t1_compressed = self.compressor.get_compressed_len(t1)
            for j, t2 in enumerate(data_to_compare):
                if fast:
                    t2_compressed = self.compressor.get_compressed_len_fast(t2)
                    t1t2_compressed = self.compressor.get_compressed_len_fast(
                        self.aggregation_func(t1, t2)
                    )
                else:
                    t2_compressed = self.compressor.get_compressed_len(t2)
                    t1t2_compressed = self.compressor.get_compressed_len(
                        self.aggregation_func(t1, t2)
                    )
                distance = self.distance_func(
                    t1_compressed, t2_compressed, t1t2_compressed
                )
                distance4i.append(distance)
            self.distance_matrix.append(distance4i)

    def calc_dis_with_single_compressed_given(
        self, data: list, data_len: list = None, train_data: list = None
    ) -> None:
        """
        Calculates the distance between either `data`, `data_len`, or `train_data`
        and appends the distance to `self.distance_matrix`.

        Arguments:
            data (list): Data to compute distance between.
            train_data (list): [Optional] Training data to compute distance from `data`.
            fast (bool): [Optional] Uses the _fast compression length function of `self.compressor`.

        Returns:
            None: None
        """

        data_to_compare = data
        if train_data is not None:
            data_to_compare = train_data

        for i, t1 in tqdm(enumerate(data)):
            distance4i = []
            t1_compressed = self.compressor.get_compressed_len_given_prob(
                t1, data_len[i]
            )
            for j, t2 in tqdm(enumerate(data_to_compare)):
                t2_compressed = self.compressor.get_compressed_len_given_prob(
                    t2, data_len[j]
                )
                t1t2_compressed = self.compressor.get_compressed_len(
                    self.aggregation_func(t1, t2)
                )
                distance = self.distance_func(
                    t1_compressed, t2_compressed, t1t2_compressed
                )
                distance4i.append(distance)
            self.distance_matrix.append(distance4i)

    def calc_dis_single(self, t1: str, t2: str) -> float:
        """
        Calculates the distance between `t1` and `t2` and returns
        that distance value as a float-like object.

        Arguments:
            t1 (str): Data 1.
            t2 (str): Data 2.

        Returns:
            float-like: Distance between `t1` and `t2`.
        """

        t1_compressed = self.compressor.get_compressed_len(t1)
        t2_compressed = self.compressor.get_compressed_len(t2)
        t1t2_compressed = self.compressor.get_compressed_len(
            self.aggregation_func(t1, t2)
        )
        distance = self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)
        return distance

    def calc_dis_single_multi(self, train_data: list, datum: str) -> list:
        """
        Calculates the distance between `train_data` and `datum` and returns
        that distance value as a float-like object.

        Arguments:
            train_data (list): Training data as a list-like object.
            datum (str): Data to compare against `train_data`.

        Returns:
            list: Distance between `t1` and `t2`.
        """

        distance4i = []
        t1_compressed = self.compressor.get_compressed_len(datum)
        for j, t2 in tqdm(enumerate(train_data)):
            t2_compressed = self.compressor.get_compressed_len(t2)
            t1t2_compressed = self.compressor.get_compressed_len(
                self.aggregation_func(datum, t2)
            )
            distance = self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)
            distance4i.append(distance)
        return distance4i

    def calc_dis_with_vector(self, data: list, train_data: list = None):
        """
        Calculates the distance between `train_data` and `data` and returns
        that distance value as a float-like object.

        Arguments:
            train_data (list): Training data as a list-like object.
            datum (str): Data to compare against `train_data`.

        Returns:
            float-like: Distance between `t1` and `t2`.
        """

        if train_data is not None:
            data_to_compare = train_data
        else:
            data_to_compare = data
        for i, t1 in tqdm(enumerate(data)):
            distance4i = []
            for j, t2 in enumerate(data_to_compare):
                distance = self.distance_func(t1, t2)
                distance4i.append(distance)
            self.distance_matrix.append(distance4i)

    def calc_acc(
        self,
        k: int,
        label: str,
        train_label: str = None,
        provided_distance_matrix: list = None,
        rand: bool = False,
    ) -> tuple:
        """
        Calculates the accuracy of the algorithm.

        Arguments:
            k (int?): TODO
            label (str): Predicted Label.
            train_label (str): Correct Label.
            provided_distance_matrix (list): Calculated Distance Matrix to use instead of `self.distance_matrix`.
            rand (bool): TODO

        Returns:
            tuple: predictions, and list of bools indicating prediction correctness.

        """
        if provided_distance_matrix is not None:
            self.distance_matrix = provided_distance_matrix
        correct = []
        pred = []
        if train_label is not None:
            compare_label = train_label
            start = 0
            end = k
        else:
            compare_label = label
            start = 1
            end = k + 1

        for i in range(len(self.distance_matrix)):
            sorted_idx = np.argsort(np.array(self.distance_matrix[i]))
            pred_labels = defaultdict(int)
            for j in range(start, end):
                pred_l = compare_label[sorted_idx[j]]
                pred_labels[pred_l] += 1
            sorted_pred_lab = sorted(
                pred_labels.items(), key=operator.itemgetter(1), reverse=True
            )
            most_count = sorted_pred_lab[0][1]
            if_right = 0
            most_label = sorted_pred_lab[0][0]
            most_voted_labels = []
            for pair in sorted_pred_lab:
                if pair[1] < most_count:
                    break
                if not rand:
                    if pair[0] == label[i]:
                        if_right = 1
                        most_label = pair[0]
                else:
                    most_voted_labels.append(pair[0])
            if rand:
                most_label = random.choice(most_voted_labels)
                if_right = 1 if most_label == label[i] else 0
            pred.append(most_label)
            correct.append(if_right)
        print("Accuracy is {}".format(sum(correct) / len(correct)))
        return pred, correct

    def combine_dis_acc(
        self,
        k: int,
        data: list,
        label: str,
        train_data: list = None,
        train_label: str = None,
    ) -> tuple:
        correct = []
        pred = []
        if train_label is not None:
            compare_label = train_label
            start = 0
            end = k
        else:
            compare_label = label
            start = 1
            end = k + 1
        if train_data is not None:
            data_to_compare = train_data
        else:
            data_to_compare = data
        for i, t1 in tqdm(enumerate(data)):
            distance4i = self.calc_dis_single_multi(data_to_compare, t1)
            sorted_idx = np.argsort(np.array(distance4i))
            pred_labels = defaultdict(int)
            for j in range(start, end):
                pred_l = compare_label[sorted_idx[j]]
                pred_labels[pred_l] += 1
            sorted_pred_lab = sorted(
                pred_labels.items(), key=operator.itemgetter(1), reverse=True
            )
            most_count = sorted_pred_lab[0][1]
            if_right = 0
            most_label = sorted_pred_lab[0][0]
            for pair in sorted_pred_lab:
                if pair[1] < most_count:
                    break
                if pair[0] == label[i]:
                    if_right = 1
                    most_label = pair[0]
            pred.append(most_label)
            correct.append(if_right)
        print("Accuracy is {}".format(sum(correct) / len(correct)))
        return pred, correct

    def combine_dis_acc_single(
        self, k: int, train_data: list, train_label: str, datum: list, label: str
    ):
        # Support multi processing - must provide train data and train label
        distance4i = self.calc_dis_single_multi(train_data, datum)
        sorted_idx = np.argpartition(np.array(distance4i), range(k))
        pred_labels = defaultdict(int)
        for j in range(k):
            pred_l = train_label[sorted_idx[j]]
            pred_labels[pred_l] += 1
        sorted_pred_lab = sorted(
            pred_labels.items(), key=operator.itemgetter(1), reverse=True
        )
        most_count = sorted_pred_lab[0][1]
        if_right = 0
        most_label = sorted_pred_lab[0][0]
        for pair in sorted_pred_lab:
            if pair[1] < most_count:
                break
            if pair[0] == label:
                if_right = 1
                most_label = pair[0]
        pred = most_label
        correct = if_right
        return pred, correct
