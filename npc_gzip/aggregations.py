import numpy as np

from npc_gzip.exceptions import StringTooShortException


def concatenate_with_space(t1: str, t2: str) -> str:
    """
    Combines `t1` and `t2` with a space.

    (formerly agg_by_concat_space)

    Arguments:
        t1 (str): First item.
        t2 (str): Second item.

    Returns:
        str: `{t1} {t2}`
    """

    return t1 + " " + t2


def aggregate_strings(stringa: str, stringb: str, by_character: bool = False) -> str:
    """
    Aggregates strings.

    (replaces agg_by_jag_char, agg_by_jag_word)

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
    i = None
    for i in range(0, minimum_list_size - 1, 2):
        combined.append(lista[i])
        combined.append(listb[i + 1])

    if i is None:
        raise StringTooShortException(t1, t2, "aggregate_strings")

    if len(lista) > len(listb):
        combined += lista[i:]

    if by_character:
        return "".join(combined)

    return " ".join(combined)


def average(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
    """
    Calculates the average of `array_a` and `array_b`,
    rounding down.

    (replaces agg_by_avg)

    Arguments:
        array_a (np.ndarray): First series of numbers.
        array_b (np.ndarray): Second series of numbers.

    Returns:
        np.ndarray: Average of the two series of numbers rounded
                    towards zero.
    """

    average_array = (a + b) / 2
    average_array = average_array.astype(int)

    return average_array


def min_max_aggregation(
    array_a: np.ndarray, array_b: np.ndarray, aggregate_by_minimum: bool = False
) -> np.ndarray:
    """
    Stacks `array_a` and `array_b` then returns the minimum value if
    `aggregate_by_minimum`, else returns the maximum value.

    Arguments:
        array_a (np.ndarray): First series of numbers.
        array_b (np.ndarray): Second series of numbers.
        aggregate_by_minimum (bool): True if you want to take the minimum of the two series.
                                     False if you want to take the maximum instead.

    Returns:
        np.ndarray: 1-D Numpy array of the min/max value from `array_a` & `array_b`.
    """

    stacked = np.stack([array_a, array_b], axis=0)
    if aggregate_by_minimum:
        return np.min(stacked, axis=0).reshape(-1)
    return torch.max(stacked, axis=0).reshape(-1)


def stack(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
    """
    Combines `array_a` and `array_b` via `np.stack`.

    Arguments:
        array_a (np.ndarray): First series of numbers.
        array_b (np.ndarray): Second series of numbers.

    Returns:
        np.ndarray: Stack of the two series.
    """

    return np.stack([array_a, array_b], axis=0)
