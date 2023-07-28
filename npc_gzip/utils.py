import random
import string

from npc_gzip.exceptions import InvalidObjectTypeException


def generate_sentence(number_of_words: int = 10) -> str:
    """
    Generates a sentence of random
    numbers and letters, with 
    `number_of_words` words in the 
    sentence such that len(out.split()) \
    == `number_of_words`.

    Arguments:
        number_of_words (int): The number of words you
                               want in the sentence.

    Returns:
        str: Sentence of random numbers and letters.
    """

    if not isinstance(number_of_words, int):
        if isinstance(number_of_words, float):
            number_of_words = int(number_of_words)
        else:
            raise InvalidObjectTypeException(
                type(number_of_words),
                supported_types=[int, float],
                function_name="generate_sentence",
            )

    assert number_of_words > 0, f"`number_of_words` must be greater than zero."

    words = []
    for word in range(number_of_words):
        # initializing size of string
        N = random.randint(1, 50)

        words.append(
            "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
        )

    out = " ".join(words)
    return out


def generate_dataset(number_of_sentences: int) -> list:
    """
    Loops over `range(number_of_sentences)` that
    utilizes `generate_sentence()` to generate
    a dataset of randomly sized sentences.

    Arguments:
        number_of_sentences (int): The number of
                                   sentences you
                                   want in your
                                   dataset.

    Returns:
        list: List of sentences (str).
    """

    if not isinstance(number_of_sentences, int):
        if isinstance(number_of_sentences, float):
            number_of_sentences = int(number_of_sentences)
        else:
            raise InvalidObjectTypeException(
                type(number_of_sentences),
                supported_types=[int, float],
                function_name="generate_dataset",
            )

    assert number_of_sentences > 0, f"`number_of_sentences` must be greater than zero."

    dataset = []
    for sentence in range(number_of_sentences):
        number_of_words = random.randint(1, 100)
        dataset.append(generate_sentence(number_of_words))

    return dataset


if __name__ == "__main__":
    number_of_sentences = random.randint(1, 100)
    dataset = generate_dataset(number_of_sentences)
    assert len(dataset) == number_of_sentences
