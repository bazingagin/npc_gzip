import numpy as np
from sklearn.metrics import classification_report
from torchtext.datasets import AG_NEWS

from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.compressors.gzip_compressor import GZipCompressor
from npc_gzip.compressors.bz2_compressor import Bz2Compressor
from npc_gzip.compressors.lzma_compressor import LzmaCompressor
from npc_gzip.knn_classifier import KnnClassifier


def get_data() -> tuple:
    """
    Pulls the AG_NEWS dataset
    and returns two tuples the first being the
    training data and the second being the test
    data. Each tuple contains the text and label
    respectively as numpy arrays.

    """

    train_iter, test_iter = AG_NEWS(split=("train", "test"))

    train_text = []
    train_labels = []
    for label, text in train_iter:
        train_labels.append(label)
        train_text.append(text)

    test_text = []
    test_labels = []
    for label, text in test_iter:
        test_labels.append(label)
        test_text.append(text)

    train_text = np.array(train_text)
    train_labels = np.array(train_labels)

    test_text = np.array(test_text)
    test_labels = np.array(test_labels)

    train = (train_text, train_labels)
    test = (test_text, test_labels)

    return (train, test)


def fit_model_gzip(
    train_text: np.ndarray, train_labels: np.ndarray, distance_metric: str = "ncd"
) -> KnnClassifier:
    """
    Fits a Knn-GZip compressor on the train
    data and returns it.

    Arguments:
        train_text (np.ndarray): Training dataset as a numpy array.
        train_labels (np.ndarray): Training labels as a numpy array.

    Returns:
        KnnClassifier: Trained Knn-Compressor model ready to make predictions.
    """

    compressor: BaseCompressor = GZipCompressor()
    model: KnnClassifier = KnnClassifier(
        compressor=compressor,
        training_inputs=train_text,
        training_labels=train_labels,
        distance_metric=distance_metric,
    )

    return model

def fit_model_bz2(
    train_text: np.ndarray, train_labels: np.ndarray, distance_metric: str = "ncd"
) -> KnnClassifier:

    compressor: BaseCompressor = Bz2Compressor()
    model: KnnClassifier = KnnClassifier(
        compressor=compressor,
        training_inputs=train_text,
        training_labels=train_labels,
        distance_metric=distance_metric,
    )

    return model

def fit_model_lzma(
    train_text: np.ndarray, train_labels: np.ndarray, distance_metric: str = "ncd"
) -> KnnClassifier:

    compressor: BaseCompressor = LzmaCompressor()
    model: KnnClassifier = KnnClassifier(
        compressor=compressor,
        training_inputs=train_text,
        training_labels=train_labels,
        distance_metric=distance_metric,
    )

    return model


def main() -> None:
    print("Fetching data...")
    ((train_text, train_labels), (test_text, test_labels)) = get_data()

    random_indicies = np.random.choice(test_text.shape[0], 1000, replace=False)

    print("Fitting model...")
    print("gzip")
    model = fit_model_gzip(train_text, train_labels)

    sample_test_text = test_text[random_indicies]
    sample_test_labels = test_labels[random_indicies]

    print("Generating predictions...")
    top_k = 1

    # Here we use the `sampling_percentage` to save time
    # at the expense of worse predictions. This
    # `sampling_percentage` selects a random % of training
    # data to compare `sample_test_text` against rather
    # than comparing it against the entire training dataset.
    (distances, labels, similar_samples) = model.predict(
        sample_test_text, top_k, sampling_percentage=0.01
    )

    print(classification_report(sample_test_labels, labels.reshape(-1)))

    
"""

    print("Fitting model...")
    print("bz2")
    model = fit_model_bz2(train_text, train_labels)

    print("Generating predictions...")

    (distances, labels, similar_samples) = model.predict(
        sample_test_text, top_k, sampling_percentage=0.01
    )

    print(classification_report(sample_test_labels, labels.reshape(-1)))\




    print("Fitting model...")
    print("lzma")
    model = fit_model_lzma(train_text, train_labels)

    print("Generating predictions...")

    (distances, labels, similar_samples) = model.predict(
        sample_test_text, top_k, sampling_percentage=0.01
    )

    print(classification_report(sample_test_labels, labels.reshape(-1)))


"""


if __name__ == "__main__":
    main()


#               precision    recall  f1-score   support

#            1       0.67      0.80      0.73       246
#            2       0.78      0.74      0.76       246
#            3       0.64      0.61      0.62       249
#            4       0.65      0.59      0.62       259

#     accuracy                           0.68      1000
#    macro avg       0.68      0.68      0.68      1000
# weighted avg       0.68      0.68      0.68      1000
