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

    stringa_list: list = stringa.split()
    stringb_list: list = stringb.split()

    zipped_lists: list = list(zip(stringa_list, stringb_list))
    out: list = list(itertools.chain(*zipped_lists))

    aggregated: str = ("" if by_character else " ").join(out)
    return aggregated
