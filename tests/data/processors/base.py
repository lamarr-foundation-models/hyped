from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from hyped.data.processors.base import BaseDataProcessor
from hyped.utils.arrow import convert_features_to_arrow_schema
from datasets import Features
from typing import Any
import pyarrow as pa
import pytest


class BaseTestDataProcessor(ABC):
    @pytest.fixture
    @abstractmethod
    def in_features(self, request) -> Features:
        ...

    @pytest.fixture
    @abstractmethod
    def processor(self, request) -> BaseDataProcessor:
        ...

    @pytest.fixture
    def batch(self, request) -> None | dict[str, list[Any]]:
        return None

    @pytest.fixture
    def expected_out_features(self, request) -> None | Features:
        return None

    @pytest.fixture
    def expected_out_batch(self, request) -> None | dict[str, list[Any]]:
        return None

    @pytest.fixture
    def expected_err_on_prepare(self) -> None | type[Exception]:
        return None

    @pytest.fixture
    def expected_err_on_process(self) -> None | type[Exception]:
        return None

    def err_handler(self, err_type) -> AbstractContextManager:
        return nullcontext() if err_type is None else pytest.raises(err_type)

    def test_case(
        self,
        in_features,
        batch,
        processor,
        expected_out_features,
        expected_out_batch,
        expected_err_on_prepare,
        expected_err_on_process,
    ):
        # check types of objects generated by fixtures
        assert isinstance(in_features, Features)
        assert isinstance(processor, BaseDataProcessor)

        # prepare and check output features
        with self.err_handler(expected_err_on_prepare):
            processor.prepare(in_features)
        # we catched an expected error
        if expected_err_on_prepare is not None:
            return

        # processor should be prepared at this point
        assert processor.is_prepared

        # check new and output features
        if expected_out_features is not None:
            assert processor.new_features == expected_out_features

            if processor.config.keep_input_features:
                assert processor.out_features == (
                    in_features | expected_out_features
                )
            else:
                assert processor.out_features == expected_out_features

        if batch is not None:
            # make sure the batch follows the feature mapping
            in_schema = convert_features_to_arrow_schema(in_features)
            table = pa.table(batch, schema=in_schema)
            # create batch index
            batch_size = len(table)
            index = list(range(batch_size))

            # apply processor
            with self.err_handler(expected_err_on_process):
                out_batch = processor.batch_process(batch, index=index, rank=0)

            if expected_err_on_process is not None:
                return

            # make sure the output batch aligns with the output features
            out_schema = convert_features_to_arrow_schema(
                processor.out_features
            )
            pa.table(out_batch, schema=out_schema)

            if expected_out_batch is not None:
                # check output content
                if processor.config.keep_input_features:
                    assert out_batch == (batch | expected_out_batch)
                else:
                    assert out_batch == expected_out_batch
