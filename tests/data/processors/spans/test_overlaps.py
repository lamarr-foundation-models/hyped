from itertools import compress

import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.spans.common import SpansOutputs
from hyped.data.processors.spans.overlaps import (
    ResolveSpanOverlaps,
    ResolveSpanOverlapsConfig,
)
from tests.data.processors.base import BaseTestDataProcessor


class TestResolveSpanOverlaps(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            [[], []],
            [[(2, 4), (5, 9)], [True, True]],
            [[(2, 5), (5, 9)], [True, True]],
            [[(2, 6), (3, 9)], [False, True]],
            [[(3, 9), (2, 6)], [False, True]],
            [[(2, 6), (3, 9), (10, 13)], [False, True, True]],
            [[(2, 6), (10, 13), (3, 9)], [False, True, True]],
            [[(2, 6), (3, 9), (10, 13), (12, 17)], [False, True, False, True]],
            [[(2, 6), (7, 9), (10, 13), (1, 17)], [True, True, True, False]],
        ],
    )
    def spans_and_mask(self, request):
        return request.param

    @pytest.fixture
    def spans(self, spans_and_mask):
        return spans_and_mask[0]

    @pytest.fixture
    def mask(self, spans_and_mask):
        return spans_and_mask[1]

    @pytest.fixture
    def in_features(self):
        return Features(
            {
                SpansOutputs.BEGINS: Sequence(Value("int32")),
                SpansOutputs.ENDS: Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def processor(self):
        return ResolveSpanOverlaps(
            ResolveSpanOverlapsConfig(
                spans_begin=SpansOutputs.BEGINS, spans_end=SpansOutputs.ENDS
            )
        )

    @pytest.fixture
    def in_batch(self, spans):
        begins, ends = ([], []) if len(spans) == 0 else zip(*spans)
        return {
            SpansOutputs.BEGINS: [list(begins)],
            SpansOutputs.ENDS: [list(ends)],
        }

    @pytest.fixture
    def expected_out_batch(self, spans, mask):
        spans = list(compress(spans, mask))
        begins, ends = ([], []) if len(spans) == 0 else zip(*spans)

        return {
            "resolve_overlaps_mask": [mask],
            SpansOutputs.BEGINS: [list(begins)],
            SpansOutputs.ENDS: [list(ends)],
        }
