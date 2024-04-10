from math import ceil
from typing import Any

from datasets import Features
from pydantic import model_validator

from hyped.common.feature_checks import (
    get_sequence_length,
    raise_feature_is_sequence,
)
from hyped.common.feature_key import FeatureKey
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


class ChunkSequenceConfig(BaseDataProcessorConfig):
    sequence: FeatureKey | list[FeatureKey]
    chunk_size: int
    chunk_stride: int = None
    drop_last: bool = False

    @model_validator(mode="after")
    def _set_default_chunk_stride(cls, config):
        if config.chunk_stride is None:
            # set default value of chunk stride
            config.chunk_stride = config.chunk_size


class ChunkSequence(BaseDataProcessor[ChunkSequenceConfig]):
    @property
    def in_feature_sequence_length(self) -> int:
        return get_sequence_length(
            self.config.sequence[0].index_features(self.in_features)
        )

    def map_features(self, features: Features) -> Features:
        chunk_features = {}
        # collect the sequence features to chunk
        for k in self.config.sequence:
            # check key type
            if not (len(k) == 1 and isinstance(k[0], str)):
                raise NotImplementedError(
                    "Chunk Sequence Processor currently only supports "
                    "simple string-like feature keys, got %s" % str(k)
                )
            # make sure the feature is a sequence
            f = k.index_features(features)
            raise_feature_is_sequence(k, f)
            # add it to the collection
            chunk_features[k[0]] = f

        # compute all sequence lengths
        lengths = list(map(get_sequence_length, chunk_features.values()))
        # make sure they match
        if any(lengths[0] != length for length in lengths[1:]):
            raise TypeError(
                "Cannot chunk along the given sequences: "
                "Lengths of the sequences mismatch, got %s" % lengths
            )

        # length match
        length = lengths[0]

        if self.config.drop_last:
            # chunks that have smaller sizes are discarded
            out_length = self.config.chunk_size

        elif (length > 0) and (
            length - self.config.chunk_size
        ) % self.config.chunk_stride == 0:
            # the sequences can be chunked perfectly and
            # each chunk has the exact chunk size
            out_length = self.config.chunk_size

        else:
            # the last chunk might have smaller size
            out_length = -1

        # set output lengths in collected features
        for f in chunk_features.values():
            f.length = out_length

        return chunk_features

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # collect all sequences to chunk over
        sequences = {k: k.index_example(example) for k in self.config.sequence}

        if self.in_feature_sequence_length == -1:
            # check if the sequence lengths of all sequences
            # match up in case they are not fixed by the features
            lengths = list(map(len, sequences.values()))
            if any(lengths[0] != length for length in lengths[1:]):
                raise TypeError(
                    "Cannot chunk along the given sequences: "
                    "Lengths of the sequences mismatch, got %s" % lengths
                )

        # compute the number of chunks for the current example
        length = len(next(iter(sequences.values())))
        num_chunks = 1 + (
            (length - self.config.chunk_size) / self.config.chunk_stride
        )
        # round up or down depending on whether the last chunk
        # should be dropped
        num_chunks = (
            int(num_chunks) if self.config.drop_last else ceil(num_chunks)
        )

        for chunk_id in range(num_chunks):
            # compute chunk borders
            chunk_start = chunk_id * self.config.chunk_stride
            chunk_end = chunk_start + self.config.chunk_size
            # yield chunk
            yield {
                k[0]: s[chunk_start:chunk_end] for k, s in sequences.items()
            }
