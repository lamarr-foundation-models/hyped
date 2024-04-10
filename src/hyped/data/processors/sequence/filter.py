from itertools import compress
from typing import Any

from datasets import ClassLabel, Features, Sequence, Value
from pydantic import Field, model_validator

from hyped.common.feature_checks import (
    get_sequence_length,
    raise_feature_is_sequence,
)
from hyped.common.feature_key import FeatureKey
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


class FilterSequenceConfig(BaseDataProcessorConfig):
    """Filter Sequence Data Processor Config

    Discard all items of a sequence that are not present
    in a specified set of valid values. The processor also
    converts the sequence feature to a `ClassLabel` instance
    which tracks ids instead of the values.

    Attributes:
        sequence (FeatureKey): the key to the sequence to filter
        valids (list[Any]): the list of valid values to keep
    """

    sequence: FeatureKey
    valids: list[Any]
    # private member for faster lookup
    valid_set: set[Any] = Field(default_factory=set, init_var=False)

    @model_validator(mode="after")
    def _construct_valid_set(cls, config):
        config.valid_set = set(config.valids)
        return config


class FilterSequence(BaseDataProcessor[FilterSequenceConfig]):
    """Filter Sequence Data Processor Config

    Discard all items of a sequence that are not present
    in a specified set of valid values. The processor also
    converts the sequence feature to a `ClassLabel` instance
    which tracks ids instead of the values.
    """

    @property
    def filtered_sequence_feature(self) -> ClassLabel:
        return self.raw_features["filtered_sequence"].feature

    def map_features(self, features: Features) -> Features:
        # check feature
        sequence = self.config.sequence.index_features(features)
        raise_feature_is_sequence(self.config.sequence, sequence)
        # get length of the original sequence
        length = get_sequence_length(sequence)
        # build output features
        return {
            "filter_mask": Sequence(Value("bool"), length=length),
            "filtered_sequence": Sequence(
                ClassLabel(names=self.config.valids)
            ),
        }

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # get the sequence from the example dict
        seq = self.config.sequence.index_example(example)
        # compute the mask and
        mask = list(map(self.config.valid_set.__contains__, seq))
        seq = self.filtered_sequence_feature.str2int(compress(seq, mask))
        # return outputs
        return {"filtered_sequence": seq, "filter_mask": mask}
