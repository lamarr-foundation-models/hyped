from itertools import chain
from typing import Any

import numpy as np
from datasets import Features
from numpy.typing import NDArray

from hyped.common.feature_checks import raise_feature_is_sequence
from hyped.data.processors.statistics.value.hist import (
    Histogram,
    HistogramConfig,
)

# TODO: write tests for sequence value histogram


class SequenceValueHistogramConfig(HistogramConfig):
    """Sequence Value Histogram Data Statistic Config

    Build a histogram of the values of a given sequence feature.

    Attributes:
        statistic_key (str):
            key under which the statistic is stored in reports.
            See `StatisticsReport` for more information.
        feature_key (FeatureKey):
            key to the dataset feature of which to build the
            histogram
        low (float): lower end of the range of the histogram
        high (float): upper end of the range of the histogram
        num_bins (int): number of bins of the histogram
    """


class SequenceValueHistogram(Histogram):
    """Sequence Value Histogram Data Statistic

    Build a histogram of the values of a given sequence feature.
    """

    # overwrite config type
    CONFIG_TYPE = SequenceValueHistogramConfig

    def check_features(self, features: Features) -> None:
        """Check input features.

        Makes sure the feature key specified in the configuration is present
        in the features and the feature type is a sequence.

        Warns when the sequence feature is fixed-sized.

        Arguments:
            features (Features): input dataset features
        """
        # make sure feature exists and is a sequence
        raise_feature_is_sequence(
            self.config.feature_key,
            self.config.feature_key.index_features(features),
        )

    def extract(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: int,
    ) -> tuple[NDArray, NDArray]:
        """Compute the sequence value histogram for the given batch
        of examples.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples
            index (list[int]): dataset indices of the batch of examples
            rank (int): execution process rank

        Returns:
            bin_ids (NDArray): array of integers containing the bin ids
            bin_counts (NDArray): array of integers containing the bin counts
        """
        x = self.config.feature_key.index_batch(examples)
        x = np.asarray(list(chain.from_iterable(x)))

        return self._compute_histogram(x)
