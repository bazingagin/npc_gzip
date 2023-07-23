import csv
import numpy as np
import math
import torch
import os
import unidecode
from collections import defaultdict, OrderedDict
from sklearn.datasets import fetch_20newsgroups

from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset


class ToInt:
    def __call__(self, pic):
        return pic * 255


def read_fn_label(fn):
    text2label = {}
    with open(fn) as fo:
        reader = csv.reader(fo, delimiter=',', quotechar='"')
        for row in reader:
            label, title, desc = row[0], row[1], row[2]
            text = '. '.join([title, desc])
            text2label[text] = label
    return text2label

def read_label(fn):
    labels = []
    with open(fn) as fo:
        reader = csv.reader(fo, delimiter=',', quotechar='"')
        for row in reader:
            label, title, desc = row[0], row[1], row[2]
            labels.append(label)
    return labels

def read_fn_compress(fn):
    text = unidecode.unidecode(open(fn).read())
    text_list = text.strip().split('\n')
    return text_list

def read_torch_text_labels(ds, indicies):
    text_list = []
    label_list = []
    for i, (label, line) in enumerate(ds):
        if i in indicies:
            text_list.append(line)
            label_list.append(label)
    return text_list, label_list

def load_20news():
    def process(d):
        pairs = []
        for i in range(len(d.data)):
            text = d.data[i]
            label = d.target[i]
            pairs.append((label, text))
        return pairs
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    train_ds, test_ds = process(newsgroups_train), process(newsgroups_test)
    return train_ds, test_ds

def load_ohsumed_single(di):
    def process(d):
        ds = []
        for dn in os.listdir(d):
            if os.path.isdir(os.path.join(d, dn)):
                label = dn
                for fn in os.listdir(os.path.join(d, dn)):
                    text = open(os.path.join(d, dn, fn)).read().strip()
                    ds.append((label, text))
        return ds
    train_dir = os.path.join(di, 'training')
    test_dir = os.path.join(di, 'test')
    train_ds, test_ds = process(train_dir), process(test_dir)
    return train_ds, test_ds

def load_ohsumed(di, split=0.9):
    train_ds = []
    test_ds = []
    for dn in os.listdir(di):
        if os.path.isdir(os.path.join(di, dn)):
            label = dn
            texts = []
            num_file = len(list(os.listdir(os.path.join(di, dn))))
            split_point = math.ceil(num_file*split)
            for i, fn in enumerate(os.listdir(os.path.join(di, dn))):
                text = open(os.path.join(di, dn, fn)).read().strip()
                texts.append(text)
                if i<split_point:
                    train_ds.append((label, text))
                else:
                    test_ds.append((label, text))
    return train_ds, test_ds

def load_r8(di, delimiter='\t'):
    def process(fn):
        l = []
        text_list = open(fn).read().strip().split('\n')
        for t in text_list:
            label, text = t.split(delimiter)
            l.append((label,text))
        return l
    test_fn = os.path.join(di, 'test.txt')
    train_fn = os.path.join(di, 'train.txt')
    train_ds, test_ds = process(train_fn), process(test_fn)
    return train_ds, test_ds

def load_trec(di):
    def process(fn):
        l = []
        with open(fn, encoding='ISO-8859-1') as fo:
            reader = csv.reader(fo, delimiter=':')
            for row in reader:
                label, text = row[0], row[1]
                l.append((label,text))
        return l
    test_fn = os.path.join(di, 'test.txt')
    train_fn = os.path.join(di, 'train.txt')
    train_ds, test_ds = process(train_fn), process(test_fn)
    return train_ds, test_ds

def load_kinnews():
    def process(ds):
        pairs = []
        for pair in ds:
            label = pair['label']
            title = pair['title']
            content = pair['content']
            pairs.append((label, title+' '+content))
        return pairs
    ds = load_dataset("kinnews_kirnews", "kinnews_cleaned")
    train_ds, test_ds = process(ds['train']), process(ds['test'])
    return train_ds, test_ds

def load_kirnews():
    def process(ds):
        pairs = []
        for pair in ds:
            label = pair['label']
            title = pair['title']
            content = pair['content']
            pairs.append((label, title+' '+content))
        return pairs
    ds = load_dataset("kinnews_kirnews", "kirnews_cleaned")
    train_ds, test_ds = process(ds['train']), process(ds['test'])
    return train_ds, test_ds

def load_swahili():
    def process(ds):
        pairs = []
        for pair in ds:
            label = pair['label']
            text = pair['text']
            pairs.append((label, text))
        return pairs
    ds = load_dataset('swahili_news')
    train_ds, test_ds = process(ds['train']), process(ds['test'])
    return train_ds, test_ds

def load_filipino():
    """deprecated - datasets on huggingface have overlapped train&test"""
    def process(ds):
        label_dict = OrderedDict()
        d = {'absent': 0, 'dengue': 1, 'health': 2, 'mosquito': 3, 'sick': 4}
        for k,v in d.items():
            label_dict[k] = v
        pairs = []
        for pair in ds:
            text = pair['text']
            for k in label_dict:
                if pair[k] == 1:
                    label = label_dict[k]
            pairs.append((label, text))
        return pairs
    ds = load_dataset('dengue_filipino')
    train_ds, test_ds = process(ds['train']), process(ds['test'])
    return train_ds, test_ds

def read_img_with_label(dataset, indicies, flatten=True):
    imgs = []
    labels = []
    for idx in indicies:
        img = np.array(dataset[idx][0])
        label = dataset[idx][1]
        if flatten:
            img = img.flatten()
        imgs.append(img)
        labels.append(label)
    return np.array(imgs), np.array(labels)

def read_img_label(dataset, indicies):
    labels = []
    for idx in indicies:
        label = dataset[idx][1]
        labels.append(label)
    return labels

def pick_n_sample_from_each_class(fn, n, idx_only=False):
    label2text = defaultdict(list)
    label2idx = defaultdict(list)
    class2count = {}
    result = []
    labels = []
    recorded_idx = []
    with open(fn) as fo:
        reader = csv.reader(fo, delimiter=',', quotechar='"')
        for i, row in enumerate(reader):
            label, title, desc = row[0], row[1], row[2]
            text = '. '.join([title, desc])
            label2text[label].append(text)
            label2idx[label].append(i)
        for cl in label2text:
            class2count[cl] = len(label2text[cl])
    for c in class2count:
        select_idx = np.random.choice(class2count[c], size=n, replace=False)
        select_text = np.array(label2text[c])[select_idx]
        select_text_idx = np.array(label2idx[c])[select_idx]
        recorded_idx += list(select_text_idx)
        result+=list(select_text)
        labels+=[c]*n
    print(len(result))
    if idx_only:
        return recorded_idx
    else:
        return result, labels

def pick_n_sample_from_each_class_given_dataset(ds, n, output_fn, index_only=False):
    label2text = defaultdict(list)
    label2idx = defaultdict(list)
    class2count = {}
    result = []
    labels = []
    recorded_idx = []
    for i, (label, text) in enumerate(ds):
        label2text[label].append(text)
        label2idx[label].append(i)
    for cl in label2text:
        class2count[cl] = len(label2text[cl])
    for c in class2count:
        select_idx = np.random.choice(class2count[c], size=n, replace=False)
        select_text = np.array(label2text[c])[select_idx]
        select_text_idx = np.array(label2idx[c])[select_idx]
        recorded_idx+=list(select_text_idx)
        result+=list(select_text)
        labels+=[c]*n
    print(len(result))
    if output_fn is not None:
        np.save(output_fn, np.array(recorded_idx))
    if index_only:
        return np.array(recorded_idx), labels
    return result, labels


def pick_n_sample_from_each_class_img(dataset, n, prefix='train', flatten=False):
    label2img = defaultdict(list)
    label2idx = defaultdict(list)
    class2count = {}
    result = []
    labels = []
    recorded_idx = [] #for replication
    for i,pair in enumerate(dataset):
        img, label = pair
        if flatten:
            img = np.array(img).flatten()
        label2img[label].append(img)
        label2idx[label].append(i)
    for cl in label2img:
        class2count[cl] = len(label2img[cl])
    for c in class2count:
        select_idx = np.random.choice(class2count[c], size=n, replace=False)
        select_img = np.array(label2img[c])[select_idx]
        select_img_idx = np.array(label2idx[c])[select_idx]
        recorded_idx+=list(select_img_idx)
        result+=list(select_img)
        labels+=[c]*n
    print(len(result))
    print(recorded_idx)
    return result, labels, recorded_idx

def load_custom_dataset(di, delimiter='\t'):
    def process(fn):
        l = []
        text_list = open(fn).read().strip().split('\n')
        for t in text_list:
            label, text = t.split(delimiter)
            l.append((label,text))
        return l
    test_fn = os.path.join(di, 'test.txt')
    train_fn = os.path.join(di, 'train.txt')
    train_ds, test_ds = process(train_fn), process(test_fn)
    return train_ds, test_ds
