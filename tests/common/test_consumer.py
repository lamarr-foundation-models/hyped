import multiprocessing as mp
from itertools import product

import datasets
import pytest
from torch.utils.data import get_worker_info

from hyped.common.consumer import (
    BaseDatasetConsumer,
    ConsumerProcessExceptionGroup,
)


def filter_false(e):
    return False


class DummyDatasetConsumer(BaseDatasetConsumer):
    def __init__(self, **kwargs):
        super(DummyDatasetConsumer, self).__init__(**kwargs)
        # keep track of total examples processed by the consumer
        # and all worker infos are set correctly
        self.total_consumed_examples = mp.Value("i", 0)
        self.total_unset_worker_infos = mp.Value("i", 0)

    def initialize_worker(self, *args, **kwargs) -> None:
        # check worker info
        if get_worker_info() is None:
            with self.total_unset_worker_infos.get_lock():
                self.total_unset_worker_infos.value += 1

    def consume_example(self, **kwargs):
        with self.total_consumed_examples.get_lock():
            self.total_consumed_examples.value += 1


class RaiseExceptionConsumer(BaseDatasetConsumer):
    def consume_example(self, **kwargs):
        raise Exception("Dummy Exception")


class TestBaseDatasetConsumer(object):
    @pytest.mark.parametrize("num_samples", [1024, 2048, 10000])
    @pytest.mark.parametrize(
        "num_proc, num_shards", product([2, 4, 8, 16], repeat=2)
    )
    def test(self, num_samples, num_proc, num_shards):
        # create dummy dataset
        ds = datasets.Dataset.from_dict({"a": range(num_samples)})
        ds = ds.to_iterable_dataset(num_shards=num_shards)
        # consume dataset
        consumer = DummyDatasetConsumer(num_proc=num_proc)
        consumer.consume(ds)
        # make sure the full dataset was consumed
        assert consumer.total_unset_worker_infos.value == 0
        assert consumer.total_consumed_examples.value == num_samples

    @pytest.mark.parametrize("num_proc", [1, 2])
    def test_forward_process_exceptions(self, num_proc):
        # create dataset and consumer
        ds = datasets.Dataset.from_dict({"a": range(10)})
        consumer = RaiseExceptionConsumer(num_proc=num_proc)
        # check that exception is fowarded to main process
        with pytest.raises(ConsumerProcessExceptionGroup) as e:
            consumer.consume(ds)
            assert len(e.value.exc_group) == num_proc

    @pytest.mark.parametrize("num_proc", [1, 2])
    def test_consume_empty_shards(self, num_proc):
        # create an empty iterable dataset
        ds = datasets.Dataset.from_dict({"a": range(10)})
        ds = ds.to_iterable_dataset(num_shards=2)
        ds = ds.filter(filter_false)
        # consume dataset
        consumer = DummyDatasetConsumer(num_proc=num_proc)
        consumer.consume(ds)
        # make sure the full dataset was consumed
        assert consumer.total_unset_worker_infos.value == 0
        assert consumer.total_consumed_examples.value == 0
