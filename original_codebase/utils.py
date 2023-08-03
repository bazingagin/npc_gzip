from collections.abc import Sequence

import numpy as np
import scipy.stats
import torch


def NCD(c1: float, c2: float, c12: float) -> float:
    """
    Calculates Normalized Compression Distance (NCD).

    Arguments:
        c1 (float): The compressed length of the first object.
        c2 (float): The compressed length of the second object.
        c12 (float): The compressed length of the concatenation of the first
                     and second objects.

    Returns:
        float: The Normalized Compression Distance c1 and c2.

    Formula:
        NCD(c1, c2, c12) = (c12 - min(c1, c2)) / max(c1, c2)
    """

    distance = (c12 - min(c1, c2)) / max(c1, c2)
    return distance


def CLM(c1, c2, c12):
    """
    Calculates Compression-based Length Measure (CLM).

    Arguments:
        c1: The compressed length of the first object.
        c2: The compressed length of the second object.
        c12: The compressed length of the concatenation of the first and second objects.

    Returns:
        float: The Compression-based Length Measure value between c1 and c2.

    Formula:
        CLM(c1, c2, c12) = 1 - (c1 + c2 - c12) / c12
    """
    dis = 1 - (c1 + c2 - c12) / c12
    return dis


def CDM(c1: float, c2: float, c12: float) -> float:
    """
    Calculates Compound Dissimilarity Measure (CDM).

    Arguments:
        c1 (float): The compressed length of the first object.
        c2 (float): The compressed length of the second object.
        c12 (float): The compressed length of the concatenation of the first
                     and second objects.

    Returns:
        float: The Compound Dissimilarity Measure value between c1 and c2.

    Formula:
        CDM(c1, c2, c12) = c12 / (c1 + c2)
    """
    dis = c12 / (c1 + c2)
    return dis


def MSE(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates Mean Squared Error (MSE).

    Arguments:
        v1 (np.ndarray): The first array.
        v2 (np.ndarray): The second array.

    Returns:
        float: The Mean Squared Error value, representing the average squared
               difference between v1 and v2.

    Formula:
        MSE(v1, v2) = Σ((v1 - v2) ** 2) / len(v1)
    """

    return np.sum((v1 - v2) ** 2) / len(v1)


def agg_by_concat_space(t1: str, t2: str) -> str:
    """
    Combines `t1` and `t2` with a space.

    Arguments:
        t1 (str): First item.
        t2 (str): Second item.

    Returns:
        str: `{t1} {t2}`
    """

    return t1 + " " + t2


def agg_by_jag_word(t1: str, t2: str) -> str:
    """
    # TODO: Better description

    Arguments:
        t1 (str): First item.
        t2 (str): Second item.

    Returns:
        str:
    """

    t1_list = t1.split(" ")
    t2_list = t2.split(" ")

    combined = []
    minimum_list_size = min([len(t1_list), len(t2_list)])
    for i in range(0, minimum_list_size - 1, 2):
        combined.append(t1_list[i])
        combined.append(t2_list[i + 1])

    if len(t1_list) > len(t2_list):
        combined += t1_list[i:]
    return " ".join(combined)


def agg_by_jag_char(t1: str, t2: str):
    """
    # TODO: Better description

    Arguments:
        t1 (str): First item.
        t2 (str): Second item.

    Returns:
        str:
    """

    t1_list = list(t1)
    t2_list = list(t2)
    combined = []
    minimum_list_size = min([len(t1_list), len(t2_list)])
    for i in range(0, minimum_list_size - 1, 2):
        combined.append(t1_list[i])
        combined.append(t2_list[i + 1])
    if len(t1_list) > len(t2_list):
        combined += t1_list[i:]

    return "".join(combined)


def aggregate_strings(stringa: str, stringb: str, by_character: bool = False) -> str:
    """
    Aggregates strings.

    Arguments:
        stringa (str): First item.
        stringb (str): Second item.
        by_character (bool): True if you want to join the combined string by character,
                             Else combines by word

    Returns:
        str: combination of stringa and stringb
    """

    lista = list(stringa)
    listb = list(stringb)
    combined = []
    minimum_list_size = min([len(lista), len(listb)])
    for i in range(0, minimum_list_size - 1, 2):
        combined.append(lista[i])
        combined.append(listb[i + 1])
    if len(lista) > len(listb):
        combined += lista[i:]

    if by_character:
        return "".join(combined)
    return " ".join(combined)


def agg_by_avg(i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the average of i1 and i2, rounding to the shortest.

    Arguments:
        i1 (torch.Tensor): First series of numbers.
        i2 (torch.Tensor): Second series of numbers.

    Returns:
        torch.Tensor: Average of the two series of numbers.
    """

    return torch.div(i1 + i2, 2, rounding_mode="trunc")


def agg_by_min_or_max(
    i1: torch.Tensor, i2: torch.Tensor, aggregate_by_minimum: bool = False
) -> torch.Tensor:
    """
    Calculates the average of i1 and i2, rounding to the shortest.

    Arguments:
        i1 (torch.Tensor): First series of numbers.
        i2 (torch.Tensor): Second series of numbers.
        aggregate_by_minimum (bool): True to take the minimum of the two series.
                                     False to take the maximum instead.

    Returns:
        torch.Tensor: Average of the two series.
    """

    stacked = torch.stack([i1, i2], axis=0)
    if aggregate_by_minimum:
        return torch.min(stacked, axis=0)[0]

    return torch.max(stacked, axis=0)[0]


def agg_by_stack(i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
    """
    Combines `i1` and `i2` via `torch.stack`.

    Arguments:
        i1 (torch.Tensor): First series of numbers.
        i2 (torch.Tensor): Second series of numbers.

    Returns:
        torch.Tensor: Stack of the two series.
    """

    return torch.stack([i1, i2])


def mean_confidence_interval(data: Sequence, confidence: float = 0.95) -> tuple:
    """
    Computes the mean confidence interval of `data` with `confidence`

    Arguments:
        data (Sequence): Data to compute a confidence interval over.
        confidence (float): Level to compute confidence.

    Returns:
        tuple: (Mean, quantile-error-size)
    """

    if isinstance(data, np.ndarray):
        array = data
    else:
        array = np.array(data, dtype=np.float32)

    n = array.shape[0]

    mean = np.mean(array)
    standard_error = scipy.stats.sem(array)
    quantile = scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, standard_error * quantile
