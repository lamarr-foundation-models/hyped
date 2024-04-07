from typing import Any

from datasets import Features
from pydantic import model_validator

from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_key import FeatureKey


class FilterFeaturesConfig(BaseDataProcessorConfig):
    """Filter Features Data Processor Config

    Removes dataset features based on the specified filters,
    i.e. the list of features to keep or remove.

    Make sure to specify exactly one of the filters.

    Type Identifier: `hyped.data.processors.features.format`

    Attributes:
        keep (None | list[FeatureKey]): features to keep
        remove (None | list[FeatureKey]): features to remove

    Raises:
        ValueError: when none of the attributes are specified
        ValueError: when both `keep` and `remove` are specified

    """

    # don't keep input features
    keep_input_features: bool = False
    # feature keys to keep or remove
    keep: None | FeatureKey | list[FeatureKey] = None
    remove: None | FeatureKey | list[FeatureKey] = None

    @model_validator(mode="after")
    def _validate_arguments(cls, config):
        keep = config.keep
        remove = config.remove

        if (keep is None) and (remove is None):
            raise ValueError(
                "No filters specified, please specify either the `keep` "
                "or `remove` filters in the configuration"
            )

        if (keep is not None) and (remove is not None):
            raise ValueError(
                "Please specify either the `keep` or the `remove` filter "
                "but not both"
            )

        return config


class FilterFeatures(BaseDataProcessor[FilterFeaturesConfig]):
    """Filter Features Data Processor

    Removes dataset features based on the specified filters,
    i.e. the list of features to keep or remove.
    """

    @property
    def required_feature_keys(self) -> list[FeatureKey]:
        # TODO: when remove is defined this should be input_features \ remove
        if self.config.remove is not None:
            raise NotImplementedError()

        return list(self.config.required_feature_keys)

    def map_features(self, features: Features) -> Features:
        """Filter dataset feature mapping

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): filtered dataset features
        """

        keep = self.config.keep
        remove = self.config.remove

        if keep is not None and isinstance(keep, FeatureKey):
            keep = [keep]

        if remove is not None and isinstance(remove, FeatureKey):
            remove = [remove]

        # make sure all features exist
        for k in keep if keep is not None else remove:
            # TODO: currently only supports string keys
            if not (len(k) == 1 and isinstance(k[0], str)):
                raise NotImplementedError(
                    "Currently only simple string keys are "
                    "supported by the filter processor"
                )

            if k[0] not in features:
                raise KeyError(
                    "`%s` not present in features, valid feature keys are %s"
                    % (k[0], list(features.keys()))
                )

        if keep is not None:
            # collect features
            return Features({k[0]: features[k[0]] for k in keep})

        if remove is not None:
            # remove features
            remove = set([k[0] for k in remove])
            return Features(
                {k: v for k, v in features.items() if k not in remove}
            )

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Filter features in example

        Arguments:
            example (dict[str, Any]): example to filter
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]): filtered example
        """
        # collect values of features to keep
        return {k: example[k] for k in self.new_features.keys()}
