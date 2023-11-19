from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from datasets import Features
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class FormatFeaturesConfig(BaseDataProcessorConfig):
    """(Re-Format) Dataset Features Processor Config

    Re-Formats Features of the dataset according to the
    specified mapping.

    Type Identifier: `hyped.data.processors.features.format`

    Attributes:
        output_format (dict[str, FeatureMappingT]):
            feature mapping describing the formatted target features,
            Leafs of the (nested) mapping must be valid feature names
            of existing dataset features or paths (i.e. tuples) in case
            of nested features.
    """

    t: Literal[
        "hyped.data.processors.features.format"
    ] = "hyped.data.processors.features.format"


class FormatFeatures(BaseDataProcessor[FormatFeaturesConfig]):
    """(Re-Format) Dataset Features Processor

    Re-Formats Features of the dataset according to the
    mapping in the config.

    Arguments:
        config (FormatFeaturesConfig): formatting configuration
    """

    def map_features(self, features: Features) -> Features:
        """Pass through features, formatting is handled by
        base data processor"""
        return features

    def internal_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """Pass through batch of examples, formatting is handled by
        base data processor"""
        return examples, list(range(len(index)))
