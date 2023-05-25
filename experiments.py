# Experiment framework
import os
import torch
import numpy as np
import statistics
import operator
from collections import Counter, defaultdict
from tqdm import tqdm
import random
from functools import partial
from itertools import repeat
from copy import deepcopy
from statistics import mode
import pickle
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score


class KnnExpText:
    def __init__(self, agg_f, comp, dis):
        self.aggregation_func = agg_f
        self.compressor = comp
        self.distance_func = dis
        self.dis_matrix = []

    def calc_dis(self, data, train_data=None, fast=False):
        if train_data is not None:
            data_to_compare = train_data
        else:
            data_to_compare = data
        for i, t1 in tqdm(enumerate(data)):
            distance4i = []
            if fast:
                t1_compressed = self.compressor.get_compressed_len_fast(t1)
            else:
                t1_compressed = self.compressor.get_compressed_len(t1)
            for j, t2 in enumerate(data_to_compare):
                if fast:
                    t2_compressed = self.compressor.get_compressed_len_fast(t2)
                    t1t2_compressed = self.compressor.get_compressed_len_fast(self.aggregation_func(t1,t2))
                else:
                    t2_compressed = self.compressor.get_compressed_len(t2)
                    t1t2_compressed = self.compressor.get_compressed_len(self.aggregation_func(t1, t2))
                distance = self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)
                distance4i.append(distance)
            self.dis_matrix.append(distance4i)

    def calc_dis_with_single_compressed_given(self, data, data_len=None, train_data=None):
        if train_data is not None:
            data_to_compare = train_data
        else:
            data_to_compare = data
        for i, t1 in tqdm(enumerate(data)):
            distance4i = []
            t1_compressed = self.compressor.get_compressed_len_given_prob(t1, data_len[i])
            for j, t2 in tqdm(enumerate(data_to_compare)):
                t2_compressed = self.compressor.get_compressed_len_given_prob(t2, data_len[j])
                t1t2_compressed = self.compressor.get_compressed_len(self.aggregation_func(t1, t2))
                distance = self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)
                distance4i.append(distance)
            self.dis_matrix.append(distance4i)

    def calc_dis_single(self, t1, t2):
        t1_compressed = self.compressor.get_compressed_len(t1)
        t2_compressed = self.compressor.get_compressed_len(t2)
        t1t2_compressed = self.compressor.get_compressed_len(self.aggregation_func(t1, t2))
        distance = self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)
        return distance
    def calc_dis_single_multi(self, train_data, datum):
        distance4i = []
        t1_compressed = self.compressor.get_compressed_len(datum)
        for j, t2 in tqdm(enumerate(train_data)):
            t2_compressed = self.compressor.get_compressed_len(t2)
            t1t2_compressed = self.compressor.get_compressed_len(self.aggregation_func(datum, t2))
            distance = self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)
            distance4i.append(distance)
        return distance4i
    def calc_dis_with_vector(self, data, train_data=None):
        if train_data is not None:
            data_to_compare = train_data
        else:
            data_to_compare = data
        for i, t1 in tqdm(enumerate(data)):
            distance4i = []
            for j, t2 in enumerate(data_to_compare):
                distance = self.distance_func(t1, t2)
                distance4i.append(distance)
            self.dis_matrix.append(distance4i)
    def calc_acc(self, k, label, train_label=None, provided_distance_matrix=None, rand=False):
        if provided_distance_matrix is not None:
            self.dis_matrix = provided_distance_matrix
        correct = []
        pred = []
        if train_label is not None:
            compare_label = train_label
            start = 0
            end = k
        else:
            compare_label = label
            start = 1
            end = k+1
        for i in range(len(self.dis_matrix)):
            sorted_idx = np.argsort(np.array(self.dis_matrix[i]))
            pred_labels = defaultdict(int)
            for j in range(start, end):
                pred_l = compare_label[sorted_idx[j]]
                pred_labels[pred_l] += 1
            sorted_pred_lab = sorted(pred_labels.items(), key=operator.itemgetter(1), reverse=True)
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
                if_right = 1 if most_label==label[i] else 0
            pred.append(most_label)
            correct.append(if_right)
        print("Accuracy is {}".format(sum(correct)/len(correct)))
        return pred, correct
    def combine_dis_acc(self, k, data, label, train_data=None, train_label=None):
        correct = []
        pred = []
        if train_label is not None:
            compare_label = train_label
            start = 0
            end = k
        else:
            compare_label = label
            start = 1
            end = k+1
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
            sorted_pred_lab = sorted(pred_labels.items(), key=operator.itemgetter(1), reverse=True)
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

    def combine_dis_acc_single(self, k, train_data, train_label, datum, label):
        # Support multi processing - must provide train data and train label
        distance4i = self.calc_dis_single_multi(train_data, datum)
        sorted_idx = np.argpartition(np.array(distance4i), range(k))
        pred_labels = defaultdict(int)
        for j in range(k):
            pred_l = train_label[sorted_idx[j]]
            pred_labels[pred_l] += 1
        sorted_pred_lab = sorted(pred_labels.items(), key=operator.itemgetter(1), reverse=True)
        most_count = sorted_pred_lab[0][1]
        if_right = 0
        most_label = sorted_pred_lab[0][0]
        for pair in sorted_pred_lab:
            if pair[1] < most_count:
                break
            if pair[0] == label:
                if_right = 1
                most_label = pair[0]
        pred=most_label
        correct=if_right
        return pred, correct


