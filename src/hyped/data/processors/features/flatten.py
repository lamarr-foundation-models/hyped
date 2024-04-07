from datasets import Features
from pydantic import Field

from hyped.data.processors.features.format import (
    FormatFeatures,
    FormatFeaturesConfig,
)
from hyped.utils.feature_key import (
    FeatureKey,
    FeatureKeyCollection,
    _iter_keys_in_features,
)


class FlattenFeaturesConfig(FormatFeaturesConfig):
    """Flatten Dataset Features Processor Config

    Similar to formatting features (see `hyped.data.processors.helpers.format`)
    but flattens nested features

    Type Identifier: `hyped.data.processors.features.flatten`

    Attributes:
        to_flatten (None | list[str]):
            dataset features to flatten. By default flattens all features
            present in the source feature mapping
        delimiter (str): delimiter used to join nested keys, defaults to ':'
        depth (int):
            when set to a positive integer, the nested structure
            of the feature mapping will only be flattened to the
            specified depth. Defaults to -1.
        max_seq_length_to_unpack (int):
            upper threshold of length to unpack sequences. If the sequence
            length exceeds this threshold, the sequence will not be unpacked
    """

    to_flatten: None | list[FeatureKey] = None
    delimiter: str = ":"
    depth: int = -1
    max_seq_length_to_unpack: int = 8

    output_format: None | FeatureKeyCollection = Field(
        default=None, init_var=False
    )


class FlattenFeatures(FormatFeatures):
    """Flatten Dataset Features Processor

    Similar to formatting features (see `hyped.data.processors.helpers.format`)
    but flattens nested features

    Arguments:
        config (None | FlattenFeaturesConfig):
            flattening configuration, defaults to flatten all features
    """

    # overwrite config type
    CONFIG_TYPE = FlattenFeaturesConfig

    def __init__(self, config: None | FlattenFeaturesConfig = None) -> None:
        super(FlattenFeatures, self).__init__(
            config=config or FlattenFeaturesConfig()
        )

    def map_features(self, features: Features) -> Features:
        """Flatten dataset feature mapping

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): flattened feature mapping
        """

        # get features to flatten, default to all features
        # in the feature mapping
        to_flatten = (
            self.config.to_flatten
            if self.config.to_flatten is not None
            else list(map(FeatureKey, features.keys()))
        )

        collection = FeatureKeyCollection.from_feature_keys(to_flatten)

        for key in to_flatten:
            # get the feature to flatten
            feature = key.index_features(features)

            # the key collection
            # to add the flattened features into
            sub_collection = key[:-1].index_example(collection)
            assert key[-1] in sub_collection
            # flatten features
            flat_collection = {
                self.config.delimiter.join(map(str, key[-1:] + k)): FeatureKey(
                    key + k
                )
                for k in map(
                    # TODO: can we flatten after slice?
                    FeatureKey.cutoff_at_slice,
                    _iter_keys_in_features(
                        feature,
                        self.config.depth,
                        self.config.max_seq_length_to_unpack,
                    ),
                )
            }

            sub_collection.pop(key[-1])
            sub_collection.update(flat_collection)

        self.config.output_format = collection
        return super(FlattenFeatures, self).map_features(features)
