import ray
import datasets
from hyped.data.pipe import DataPipe
from hyped.data.dist.pipe import DistributedDataPipe
from hyped.data.dist.parallel import DistributedParallelDataPipe
from tests.data.processors.test_base import (
    ConstantDataProcessor,
    ConstantDataProcessorConfig,
)
from .test_pipe import TestDistributedDataPipe as _TestDistributedDataPipe
import pytest


class TestDistributedParallelDataPipe(_TestDistributedDataPipe):
    @pytest.fixture(
        params=[
            "parallel-A",
            "parallel-B",
            "parallel-C",
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
        if request.param == "parallel-A":
            return DistributedDataPipe(
                [
                    p1,
                    DistributedParallelDataPipe(
                        [
                            DistributedDataPipe([p2]),
                            DistributedDataPipe([p3]),
                        ]
                    ),
                ]
            )
        if request.param == "parallel-B":
            return DistributedDataPipe(
                [
                    DistributedParallelDataPipe(
                        [
                            DistributedDataPipe([p1]),
                            DistributedDataPipe([p2]),
                            DistributedDataPipe([p3]),
                        ]
                    )
                ]
            )
        if request.param == "parallel-C":
            return DistributedDataPipe(
                [
                    DistributedParallelDataPipe(
                        [
                            DistributedDataPipe([p1]),
                            DistributedDataPipe([p2]),
                        ]
                    ),
                    p3,
                ]
            )

        raise ValueError(request.param)
