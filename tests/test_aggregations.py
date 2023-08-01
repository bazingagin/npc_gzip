from npc_gzip.aggregations import aggregate_strings, concatenate_with_space


class TestAggregations:
    stringa: str = "hey there how are you?"
    stringb: str = "I am just hanging out!"

    def test_concatenate_with_space(self) -> None:
        out = concatenate_with_space(self.stringa, self.stringb)

        assert len(out) == len(self.stringa) + len(self.stringb) + 1
        assert out == f"{self.stringa} {self.stringb}"

    def test_aggregate_strings(self) -> None:
        out = aggregate_strings(self.stringa, self.stringb, by_character=False)
        assert len(out) == len(self.stringa) + len(self.stringb) + 1

    def test_aggregate_strings_by_character(self) -> None:
        out = aggregate_strings(self.stringa, self.stringb, by_character=True)
        total_length = len("".join(self.stringa.split()))
        total_length += len("".join(self.stringb.split()))
        assert len(out) == total_length
