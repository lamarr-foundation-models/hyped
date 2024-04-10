import warnings
from typing import Any

import numpy as np
from datasets import Features
from numpy.typing import NDArray
from pydantic import Field

from hyped.common.feature_checks import (
    get_sequence_length,
    raise_feature_is_sequence,
)
from hyped.data.processors.statistics.value.hist import (
    Histogram,
    HistogramConfig,
)

# TODO: write tests for sequence length histogram


class SequenceLengthHistogramConfig(HistogramConfig):
    """Sequence Length Histogram Data Statistic Config

    Build a histogram over the lengths of a given sequence feature.

    Attributes:
        statistic_key (str):
            key under which the statistic is stored in reports.
            See `StatisticsReport` for more information.
        feature_key (FeatureKey):
            key to the dataset feature of which to build the
            histogram
        max_length (int):
            maximum sequence length, sequences exceeding this
            threshold will be truncated for the statistics computation
    """

    max_length: int

    low: float = Field(default=0, init_var=False)
    high: float = Field(default=0, init_var=False)
    num_bins: float = Field(default=0, init_var=False)

    def model_post_init(self, __context) -> None:
        # set values
        self.low = 0
        self.high = self.num_bins = self.max_length

        super(SequenceLengthHistogramConfig, self).__model_post_init__(
            __context
        )


class SequenceLengthHistogram(Histogram):
    """Sequence Length Histogram Data Statistic

    Build a histogram over the lengths of a given sequence feature.
    """

    # overwrite config type
    CONFIG_TYPE = SequenceLengthHistogramConfig

    def check_features(self, features: Features) -> None:
        """Check input features.

        Makes sure the feature key specified in the configuration is present
        in the features and the feature type is a sequence.

        Warns when the sequence feature is fixed-sized.

        Arguments:
            features (Features): input dataset features
        """
        # make sure feature exists and is a sequence
        feature = self.config.feature_key.index_features(features)
        raise_feature_is_sequence(self.config.feature_key, feature)
        # warn about fixed length sequences
        if get_sequence_length(feature) != -1:
            warnings.warn(
                "Computing sequence length histogram of fixed length sequence",
                UserWarning,
            )

    def extract(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: int,
    ) -> tuple[NDArray, NDArray]:
        """Compute the sequence length histogram for the given
        batch of examples.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples
            index (list[int]): dataset indices of the batch of examples
            rank (int): execution process rank

        Returns:
            bin_ids (NDArray): array of integers containing the bin ids
            bin_counts (NDArray): array of integers containing the bin counts
        """
        x = self.config.feature_key.index_batch(examples)
        lengths = list(map(len, x))

        return self._compute_histogram(np.asarray(lengths))
