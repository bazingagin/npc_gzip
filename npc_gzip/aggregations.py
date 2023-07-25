import torch
from npc_gzip.exceptions import StringTooShortException

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

    i = None
    combined = []
    minimum_list_size = min([len(t1_list), len(t2_list)])
    for i in range(0, minimum_list_size - 1, 2):
        combined.append(t1_list[i])
        combined.append(t2_list[i + 1])
    if i is None:
        raise StringTooShortException(t1, t2, 'agg_by_jag_word')
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
        aggregate_by_minimum (bool): True if you want to take the minimum of the two series.
                                     False if you want to take the maximum instead.

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
