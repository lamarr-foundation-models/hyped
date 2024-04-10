from itertools import chain
from typing import Any

from datasets import Features, Sequence

from hyped.common.feature_checks import (
    get_sequence_feature,
    get_sequence_length,
    raise_feature_is_sequence,
    raise_object_matches_feature,
)
from hyped.common.feature_key import FeatureKey
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


class ExtendSequenceConfig(BaseDataProcessorConfig):
    """Extend Sequence Data Processor Config

    Extend a sequence feature by appending or prepending
    new values.

    Arguments:
        sequence (FeatureKey): the key to the sequence to extend
        output (str): the output feature name
        append (list[Any]): values to append to the sequence
        prepend (list[Any]): values to prepend to the sequence
    """

    sequence: FeatureKey
    output: str = "output"
    append: list[Any] = []
    prepend: list[Any] = []


class ExtendSequence(BaseDataProcessor[ExtendSequenceConfig]):
    """Extend Sequence Data Processor

    Extend a sequence feature by appending or prepending
    new values.
    """

    def map_features(self, features: Features) -> Features:
        # check feature
        sequence = self.config.sequence.index_features(features)
        raise_feature_is_sequence(self.config.sequence, sequence)
        # get item feature type and length of the sequence
        feature = get_sequence_feature(sequence)
        length = get_sequence_length(sequence)
        # make sure append and prepend values match the feature type
        raise_object_matches_feature(self.config.prepend, Sequence(feature))
        raise_object_matches_feature(self.config.append, Sequence(feature))

        # compute the new sequence length
        if length != -1:
            length += len(self.config.prepend)
            length += len(self.config.append)

        # overwrite sequence feature with new sequence feature with
        # potential new length
        return Features({self.config.output: Sequence(feature, length=length)})

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # get sequence and add new values
        sequence = self.config.sequence.index_example(example)
        sequence = chain(self.config.prepend, sequence, self.config.append)
        # return updated sequence
        return {self.config.output: list(sequence)}
