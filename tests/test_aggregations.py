import pytest
from npc_gzip.exceptions import StringTooShortException
from npc_gzip.aggregations import (
    concatenate_with_space,
    aggregate_strings,
    average,
    min_max_aggregation,
    stack
)

class TestAggregations:

    stringa: str = 'hey there how are you?'
    stringb: str = 'I am just hanging out!'

    def test_concatenate_with_space(self):
        out = concatenate_with_space(self.stringa, self.stringb)
        
        assert len(out) == len(self.stringa) + len(self.stringb) + 1
        assert out == f'{self.stringa} {self.stringb}'
        
    def test_aggregate_strings(self):
        out = aggregate_strings(
            self.stringa, self.stringb, by_character=False
        )
        assert len(out) == len(self.stringa) + len(self.stringb) + 1

    def test_aggregate_strings_by_character(self):
        out = aggregate_strings(
            self.stringa, self.stringb, by_character=True
        )
        total_length = len(''.join(self.stringa.split()))
        total_length += len(''.join(self.stringb.split()))
        assert len(out) == total_length
