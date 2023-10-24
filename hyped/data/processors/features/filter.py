from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from datasets import Features
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class FilterFeaturesConfig(BaseDataProcessorConfig):
    """Filter Features Data Processor Config

    Removes dataset features based on the specified filters,
    i.e. the list of features to keep or remove.

    Type Identifier: `hyped.data.processors.features.format`

    Attributes:
        keep (None | str | list[str]): features to keep
        remove (None | str | list[str]): features to remove
    """

    t: Literal[
        "hyped.data.processors.features.filter"
    ] = "hyped.data.processors.features.filter"
    # don't keep input features
    keep_input_features: bool = False
    # feature keys to keep or remove
    keep: None | str | list[str] = None
    remove: None | str | list[str] = None


class FilterFeatures(BaseDataProcessor[FilterFeaturesConfig]):
    """Filter Features Data Processor

    Removes dataset features based on the specified filters,
    i.e. the list of features to keep or remove.
    """

    def map_features(self, features: Features) -> Features:
        """Filter dataset feature mapping

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): filtered dataset features
        """

        keep = self.config.keep
        remove = self.config.remove

        # check configuration
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

        if keep is not None:
            keep = [keep] if isinstance(keep, str) else keep

        if remove is not None:
            remove = [remove] if isinstance(remove, str) else remove

        # make sure all features are present
        for k in keep if keep is not None else remove:
            if k not in features:
                raise KeyError(
                    "Key `%s` not present in feature mapping, valid "
                    "keys are %s" % (k, list(features.keys()))
                )

        if keep is not None:
            # collect features
            return Features({k: features[k] for k in keep})

        if remove is not None:
            # remove features
            remove = set(remove)
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
