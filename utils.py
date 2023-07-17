import numpy as np
import scipy
import torch


def NCD(c1: float, c2: float, c12: float) -> float:
    """
    Calculates Normalized Compression Distance.

    """

    distance = (c12 - min(c1, c2)) / max(c1, c2)
    return distance


def CLM(c1, c2, c12):
    dis = 1 - (c1 + c2 - c12) / c12
    return dis


def CDM(c1: float, c2: float, c12: float) -> float:
    """
    Calculates Compound Dissimilarity Measure.

    """
    dis = c12 / (c1 + c2)
    return dis


def MSE(v1: float, v2: float) ->float:
    """
    Calculates Mean Squared Error.
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

def agg_by_avg(i1: list, i2: list) -> torch.Tensor:
    """
    Calculates the average of i1 and i2 rounding to the
    shortest list.

    Arguments:
        i1 (list): List 1.
        i2 (list): List 2.
    
    Returns:
        torch.Tensor: Average of the two lists.
    """

    return torch.div(i1 + i2, 2, rounding_mode="trunc")


def agg_by_min_or_max(i1: list, i2: list, aggregate_by_minimum: bool = False) -> torch.Tensor:
    """
    Calculates the average of i1 and i2 rounding to the
    shortest list.

    Arguments:
        i1 (list): List 1.
        i2 (list): List 2.
        aggregate_by_minimum (bool): True if you want to take the minimum of the two lists.
                                     False if you want to take the maximum instead.
    
    Returns:
        torch.Tensor: Average of the two lists.
    """
    
    stacked = torch.stack([i1, i2], axis=0)
    if aggregate_by_minimum:
        return torch.min(stacked, axis=0)[0]
    
    return torch.max(stacked, axis=0)[0]


def agg_by_stack(i1: list, i2: list) -> torch.Tensor:
    """
    Combines `i1` and `i2` via `torch.stack`.
    
    Arguments:
        i1 (list): List 1.
        i2 (list): List 2.
    
    Returns:
        torch.Tensor: Stack of the two lists.
    """
    
    return torch.stack([i1, i2])


def mean_confidence_interval(data: list, confidence: float = 0.95) -> tuple:
    """
    Computes the mean confidence interval of `data` with `confidence`
    
    Arguments:
        data (list): List of data to compute a confidence interval over.
        confidence (float): Level to compute confidence.

    Returns:
        tuple: (Mean, quantile-error-size)
    """

    if isinstance(data, list):
        data = np.array(data, dtype=np.float32)

    n = data.shape[0]

    mean = np.mean(data)
    standard_error = scipy.stats.sem(data)
    quantile = scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, standard_error * quantile
