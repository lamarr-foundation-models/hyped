from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    INDEX_TYPES,
    raise_feature_exists,
    raise_feature_is_sequence,
    raise_features_align,
    get_sequence_length,
    get_sequence_feature,
)
from hyped.utils.spans import (
    make_spans_exclusive,
)
import numpy as np
from enum import StrEnum
from dataclasses import dataclass
from datasets import Features, Sequence, ClassLabel, Value
from typing import Any, Literal


class BioTaggerOutputs(StrEnum):
    """Enumeration of outputs of the bio tagger processor"""

    BIO_TAGS = "bio_tags"
    """Output column containing the generated bio tag sequence"""


@dataclass
class BioTaggerConfig(BaseDataProcessorConfig):
    """Begin-In-Out (BIO) Tagger Config

    Convert Entity span annotations to per-token labels
    using the BIO-tagging scheme.

    Attributes:
        begin_tag_prefix (str):
            tag prefix used to mark the beginning of a new entity
            of a specific class
        in_tag_prefix (str):
            tag prefix used to mark the interior of an entity
            of a specific class
        out_tag (str):
            out tag used to mark tokens that are not part of any entity
        input_sequence (str): column containing the input sequence
        entity_spans_begin (str):
            column containing begins of the entity span annotations
        entity_spans_end (str):
            column containing ends of the entity span annotations
        entity_spans_label (str):
            column containing the entity class label to each entity
            span
        entity_spans_inclusive (bool):
            whether the end coordinate of the entity spans are
            inclusive or exclusive. Defaults to false.
    """

    t: Literal[
        "hyped.data.processors.taggers.bio"
    ] = "hyped.data.processors.taggers.bio"

    begin_tag_prefix: str = "B-"
    in_tag_prefix: str = "I-"
    out_tag: str = "O"

    input_sequence: str = None
    entity_spans_begin: str = None
    entity_spans_end: str = None
    entity_spans_label: str = None

    entity_spans_inclusive: bool = False


class BioTagger(BaseDataProcessor[BioTaggerConfig]):
    """Begin-In-Out (BIO) Tagger Config

    Convert Entity span annotations to per-token labels
    using the BIO-tagging scheme.
    """

    @property
    def entity_label_space(self) -> None | ClassLabel:
        """Entity label-space extracted from input features"""
        feature = self.in_features[self.config.entity_spans_label]
        feature = get_sequence_feature(feature)
        return feature if isinstance(feature, ClassLabel) else None

    @property
    def bio_label_space(self) -> None | ClassLabel:
        """Bio tags label-space extracted from new features"""
        feature = self.raw_features[BioTaggerOutputs.BIO_TAGS]
        feature = get_sequence_feature(feature)
        return feature if isinstance(feature, ClassLabel) else None

    def _tag_sequence_feature(self, features: Features) -> Sequence:
        """Build the tag sequence dataset feature given the input
        feature mapping

        If the entity labels feature is a sequence of class labels, then
        the bio tag label-space is inferred from it by applying the BIO
        label scheme. Otherwise the tag sequence will be a sequence of
        strings.

        Arguments:
            features (Features): input dataset features

        Returns:
            tag_seq (Sequence): the dataset feature for the bio tags
        """
        # the entity class label feature must be a sequence of
        # string values or class labels
        raise_feature_is_sequence(
            self.config.entity_spans_label,
            features[self.config.entity_spans_label],
            [Value("string"), ClassLabel],
        )

        # get the item feature type and length of the sequence
        feature = get_sequence_feature(
            features[self.config.entity_spans_label]
        )
        length = get_sequence_length(features[self.config.input_sequence])

        # build output feature type
        if isinstance(feature, ClassLabel):
            bio_feature_type = ClassLabel(
                names=[self.config.out_tag]
                + [
                    "%s%s" % (prefix, label)
                    for label in feature.names
                    for prefix in [
                        self.config.begin_tag_prefix,
                        self.config.in_tag_prefix,
                    ]
                ]
            )

            return Sequence(bio_feature_type, length=length)

        # otherwise the input feature type must be string
        # in which case keep it a string
        return Sequence(Value("string"), length=length)

    def map_features(self, features: Features) -> Features:
        """Check input features and return feature mapping
        for the bio tags.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): bio tags feature mapping
        """
        # make sure the input sequence exists and is a sequence
        raise_feature_exists(self.config.input_sequence, features)
        raise_feature_is_sequence(
            self.config.input_sequence, features[self.config.input_sequence]
        )

        # make sure entity spans exist and are of correct type
        raise_feature_exists(self.config.entity_spans_begin, features)
        raise_feature_exists(self.config.entity_spans_end, features)
        # begin and end sequences should contain indices
        raise_feature_is_sequence(
            self.config.entity_spans_begin,
            features[self.config.entity_spans_begin],
            INDEX_TYPES,
        )
        raise_feature_is_sequence(
            self.config.entity_spans_end,
            features[self.config.entity_spans_end],
            INDEX_TYPES,
        )
        raise_features_align(
            self.config.entity_spans_begin,
            self.config.entity_spans_end,
            features[self.config.entity_spans_begin],
            features[self.config.entity_spans_end],
        )

        return {
            BioTaggerOutputs.BIO_TAGS: self._tag_sequence_feature(features)
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
            out (dict[str, Any]): token-level span annotations
        """
        # get length of input sequence
        length = len(example[self.config.input_sequence])

        # get entity spans
        spans = zip(
            example[self.config.entity_spans_begin],
            example[self.config.entity_spans_end],
        )
        # make entity spans exclusive and filter overlapping spans
        spans = make_spans_exclusive(spans, self.config.entity_spans_inclusive)

        # get the entity labels
        labels = example[self.config.entity_spans_label]
        # convert label ids to label strings
        if self.entity_label_space is not None:
            labels = self.entity_label_space.int2str(labels)

        # build initial tag sequence of all out tags
        tags = np.full(length, fill_value=self.config.out_tag, dtype=object)

        # insert all entity spans
        for label, (b, e) in zip(labels, spans):
            # check for overlaps with previous annotations
            if (tags[b:e] != self.config.out_tag).any():
                # get the overlapping entity types
                overlap_types = [label] + [
                    (
                        tag.removeprefix(
                            self.config.begin_tag_prefix
                        ).removeprefix(self.config.in_tag_prefix)
                    )
                    for tag in tags[b:e]
                    if tag != self.config.out_tag
                ]
                # raise error on overlap
                raise ValueError(
                    "Detected overlap between entities of types %s"
                    % ", ".join(overlap_types)
                )

            # add entity to tag sequence
            tags[b:e] = "%s%s" % (self.config.in_tag_prefix, label)
            tags[b] = "%s%s" % (self.config.begin_tag_prefix, label)

        # convert numpy array to list
        tags = tags.tolist()
        # convert label strings to label ids
        if self.bio_label_space is not None:
            tags = self.bio_label_space.str2int(tags)

        # return bio tags
        return {BioTaggerOutputs.BIO_TAGS: tags}
