import ray
import datasets
from hyped.data.pipe import DataPipe
from hyped.data.dist.pipe import DistributedDataPipe
from tests.data.processors.test_base import (
    ConstantDataProcessor,
    ConstantDataProcessorConfig,
)
from tests.data.test_pipe import TestDataPipe
import pytest


class TestDistributedDataPipe(TestDataPipe):
    @pytest.fixture(scope="class", autouse=True)
    def initialize_ray(self):
        ray.init()
        yield
        ray.shutdown()

    @pytest.fixture(
        params=[
            "flat",
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

    @pytest.mark.skip(reason="distributed data pipe needs to be initialized")
    def test_batch_processing(self):
        pass

    @pytest.mark.skip(reason="NotImplemented")
    def test_apply_to_iterable_dataset_dict(self):
        pass

    @pytest.mark.skip(reason="NotImplemented")
    def test_apply_to_iterable_dataset(self):
        pass
