import random

import pytest

from npc_gzip.exceptions import InvalidObjectTypeException
from npc_gzip.utils import generate_dataset, generate_sentence


class TestUtils:
    def test_generate_sentence(self):
        for _ in range(100):
            number_of_words = random.randint(1, 100)
            sentence = generate_sentence(number_of_words)
            assert len(sentence.split()) == number_of_words

        with pytest.raises(InvalidObjectTypeException):
            generate_sentence("hey there")

        with pytest.raises(AssertionError):
            generate_sentence(-1)

    def test_generate_dataset(self):
        for _ in range(100):
            number_of_sentences = random.randint(1, 100)
            dataset = generate_dataset(number_of_sentences)
            assert len(dataset) == number_of_sentences

        with pytest.raises(InvalidObjectTypeException):
            generate_dataset("hey there")

        with pytest.raises(AssertionError):
            generate_dataset(-1)
