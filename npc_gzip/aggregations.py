import numpy as np
import itertools


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

    stringa = str(stringa)
    stringb = str(stringb)

    stringa_list: list = stringa.split()
    stringb_list: list = stringb.split()

    zipped_lists: list = list(zip(stringa_list, stringb_list))
    out: list = list(itertools.chain(*zipped_lists))

    aggregated: str = " ".join(out)
    if by_character:
        aggregated: str = "".join(out)

    return aggregated


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
