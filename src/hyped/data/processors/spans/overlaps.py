from itertools import compress
from typing import Any

from datasets import Features, Sequence, Value

from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    INDEX_TYPES,
    get_sequence_feature,
    get_sequence_length,
    raise_feature_is_sequence,
    raise_features_align,
)
from hyped.utils.feature_key import FeatureKey
from hyped.utils.spans import (
    ResolveOverlapsStrategy,
    make_spans_exclusive,
    resolve_overlaps,
)

from .outputs import SpansOutputs


class ResolveSpanOverlapsConfig(BaseDataProcessorConfig):
    """Resolve Span Overlaps Data Processor Config

    Resolve overlaps between spans of a span sequence.

    Attributes:
        spans_begin (FeatureKey):
            input feature containing the begin values of the span sequence A.
        spans_end (FeatureKey):
            input feature containing the end values of the span sequence A.
        is_spans_inclusive (bool):
            whether the end coordinates of the spans in the sequence A are
            inclusive or exclusive. Defaults to false.
        strategy (ResolveOverlapsStrategy):
            the strategy to apply when resolving the overlaps. Defaults to
            `ResolveOverlapsStrategy.APPROX` which aims to minimize the
            number of spans to remove. For other options please refer to
            `hyped.utils.spans.ResolveOverlapsStrategy`.
    """

    # span sequence
    spans_begin: FeatureKey
    spans_end: FeatureKey
    is_spans_inclusive: bool = False
    # strategy to apply
    strategy: ResolveOverlapsStrategy = ResolveOverlapsStrategy.APPROX


class ResolveSpanOverlaps(BaseDataProcessor[ResolveSpanOverlapsConfig]):
    """Resolve Span Overlaps Data Processor

    Resolve overlaps between spans of a span sequence.
    """

    def map_features(self, features: Features) -> Features:
        """Check input features and overwrite the given
        span sequence. Also returns a mask over the initial span
        sequence indicating which spans of the sequence where kept.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): output feature mapping
        """
        # make sure all features exist
        spans_begin = self.config.spans_begin.index_features(features)
        spans_end = self.config.spans_end.index_features(features)
        # spans must be sequence of integers
        raise_feature_is_sequence(
            self.config.spans_begin,
            spans_begin,
            INDEX_TYPES,
        )
        raise_feature_is_sequence(
            self.config.spans_begin,
            spans_end,
            INDEX_TYPES,
        )
        # and they must align excatly
        raise_features_align(
            self.config.spans_begin,
            self.config.spans_end,
            spans_begin,
            spans_end,
        )
        # get item feature and length from span sequence feature
        feature = get_sequence_feature(spans_begin)
        length = get_sequence_length(spans_begin)
        # returns a mask over the span sequence and overwrite
        # the span sequence
        return {
            "resolve_overlaps_mask": Sequence(Value("bool"), length=length),
            SpansOutputs.BEGINS: Sequence(feature),
            SpansOutputs.ENDS: Sequence(feature),
        }

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Apply processor to an example

        Arguments:
            example (dict[str, Any]): example to process
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]): spans without overlaps
        """

        spans = list(
            zip(
                self.config.spans_begin.index_example(example),
                self.config.spans_end.index_example(example),
            )
        )

        if len(spans) == 0:
            # handle edgecase no spans
            return {
                "resolve_overlaps_mask": [],
                SpansOutputs.BEGINS: [],
                SpansOutputs.ENDS: [],
            }

        # make spans exclusive and resolve overlaps
        excl_spans = make_spans_exclusive(
            spans, self.config.is_spans_inclusive
        )
        mask = resolve_overlaps(excl_spans, strategy=self.config.strategy)
        # apply mask to spans and return features
        spans = compress(spans, mask)
        spans_begin, spans_end = zip(*spans)

        return {
            "resolve_overlaps_mask": mask,
            SpansOutputs.BEGINS: list(spans_begin),
            SpansOutputs.ENDS: list(spans_end),
        }
