from itertools import chain
from typing import Any

from datasets import ClassLabel, Features, Value

from hyped.data.processors.statistics.value.disc_hist import (
    DiscreteHistogram,
    DiscreteHistogramConfig,
)
from hyped.utils.feature_checks import (
    INT_TYPES,
    UINT_TYPES,
    check_feature_is_sequence,
    raise_feature_is_sequence,
)

# TODO: write tests for sequence value histogram


class DiscreteSequenceValueHistogramConfig(DiscreteHistogramConfig):
    """Discrete Sequence Value Histogram Data Statistic Config

    Build a histogram of a given discrete sequence feature,
    e.g. ClassLabel or string.

    Attributes:
        statistic_key (str):
            key under which the statistic is stored in reports.
            See `StatisticsReport` for more information.
        feature_key (FeatureKey):
            key to the dataset feature of which to build the
            histogram
    """


class DiscreteSequenceValueHistogram(DiscreteHistogram):
    """Discrete Sequence Value Histogram Data Statistic

    Build a histogram of a given discrete sequence feature,
    e.g. ClassLabel or string.
    """

    # overwrite config type
    CONFIG_TYPE = DiscreteSequenceValueHistogramConfig

    def _map_values(self, vals: list[Any]) -> list[Any]:
        # get feature
        feature = self.config.feature_key.index_features(self.in_features)
        # check if feature is a class label
        if check_feature_is_sequence(feature, ClassLabel):
            # map class ids to names
            return feature.feature.int2str(vals)

        return vals

    def check_features(self, features: Features) -> None:
        """Check input features.

        Makes sure the feature key specified in the configuration is present
        in the features and the feature type is a sequence of discrete values.

        Arguments:
            features (Features): input dataset features
        """
        # make sure feature exists and is a sequence
        raise_feature_is_sequence(
            self.config.feature_key,
            self.config.feature_key.index_features(features),
            INT_TYPES + UINT_TYPES + [ClassLabel, Value("string")],
        )

    def extract(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: int,
    ) -> dict[Any, int]:
        """Compute the histogram for the given batch of examples.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples
            index (list[int]): dataset indices of the batch of examples
            rank (int): execution process rank

        Returns:
            hist (dict[Any, str]): histogram of given batch of examples
        """
        x = self.config.feature_key.index_batch(examples)
        x = list(chain.from_iterable(x))

        return self._compute_histogram(x)
