import ray
import datasets
from hyped.data.pipe import DataPipe
from hyped.data.dist.pipe import DistributedDataPipe
from tests.data.processors.test_base import (
    ConstantDataProcessor,
    ConstantDataProcessorConfig,
)
from tests.data.test_pipe import TestDataPipe as _TestDataPipe
import pytest


class TestDistributedDataPipe(_TestDataPipe):
    @pytest.fixture(scope="class", autouse=True)
    def initialize_ray(self):
        ray.init()
        yield
        ray.shutdown()

    @pytest.fixture(
        params=[
            "flat",
            "stacked",
            "stacked-1-1-1",
            "stacked-1-3-5",
            "nested",
            "nested-1-3",
            "nested-dist-and-non-dist",
        ]
    )
    def sample_data_pipe(self, request):
        # create data processor configs
        c1 = ConstantDataProcessorConfig(name="A", value="1")
        c2 = ConstantDataProcessorConfig(name="B", value="2")
        c3 = ConstantDataProcessorConfig(name="C", value="3")
        # create data processors
        p1 = ConstantDataProcessor(c1)
        p2 = ConstantDataProcessor(c2)
        p3 = ConstantDataProcessor(c3)
        # create data pipe
        if request.param == "flat":
            return DistributedDataPipe([p1, p2, p3])
        if request.param == "stacked":
            return DistributedDataPipe(
                [
                    DistributedDataPipe([p1]),
                    DistributedDataPipe([p2]),
                    DistributedDataPipe([p3]),
                ],
            )
        if request.param == "stacked-1-1-1":
            return DistributedDataPipe(
                [
                    DistributedDataPipe([p1], num_proc=1),
                    DistributedDataPipe([p2], num_proc=1),
                    DistributedDataPipe([p3], num_proc=1),
                ],
            )
        if request.param == "stacked-1-3-5":
            return DistributedDataPipe(
                [
                    DistributedDataPipe([p1], num_proc=1),
                    DistributedDataPipe([p2], num_proc=3),
                    DistributedDataPipe([p3], num_proc=5),
                ],
            )
        if request.param == "nested":
            return DistributedDataPipe(
                [p1, DistributedDataPipe([p2, DistributedDataPipe([p3])])]
            )
        if request.param == "nested-1-3":
            return DistributedDataPipe(
                [
                    p1,
                    DistributedDataPipe(
                        [p2, DistributedDataPipe([p3], num_proc=1)], num_proc=3
                    ),
                ]
            )
        if request.param == "nested-dist-and-non-dist":
            return DistributedDataPipe(
                [
                    p1,
                    DataPipe(
                        [p2, DistributedDataPipe([p3])],
                    ),
                ]
            )

        raise TypeError(request.param)

    def test_preparation_logic(self, sample_data_pipe):
        sample_data_pipe._spawn_pool(num_actors=1)
        super(TestDistributedDataPipe, self).test_preparation_logic(
            sample_data_pipe
        )

    def test_feature_management(self, sample_data_pipe):
        sample_data_pipe._spawn_pool(num_actors=1)
        super(TestDistributedDataPipe, self).test_feature_management(
            sample_data_pipe
        )

    def test_batch_processing(self, sample_data_pipe):
        sample_data_pipe._spawn_pool(num_actors=1)
        super(TestDistributedDataPipe, self).test_batch_processing(
            sample_data_pipe
        )

    @pytest.mark.skip(reason="NotImplementedError")
    def test_apply_to_iterable_dataset(self):
        pass

    @pytest.mark.skip(reason="NotImplementedError")
    def test_apply_to_iterable_dataset_dict(self):
        pass
