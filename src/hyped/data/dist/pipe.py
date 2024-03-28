from __future__ import annotations
import ray
import datasets
from ray.actor import ActorHandle
from queue import Queue
from torch.utils.data import get_worker_info
from multiprocessing.managers import BaseManager
from typing import Any, Iterable

from hyped.data.pipe import DataPipe
from hyped.data.processors.base import BaseDataProcessor
from hyped.utils.feature_checks import check_feature_equals


class RemoteDataPipe(DataPipe):
    """(Internal) Remote Data Pipe

    Class that is distributed internally by `DistributedDataPipe`.
    Provides helper functions for distributed setting on top of
    the standard `DataPipe` functionality.
    """

    def _self(self, attr_name: str) -> Any:
        return getattr(self, attr_name)


# TODO: remote data pipes currently do not support statistics
#       processors, this requires the statistics to be send
#       from the remote actor to the main process
class DistributedDataPipe(DataPipe):
    """Distributed Data Pipe

    Uses `ray` to distribute the heavy workload over a ray cluster.
    It creates a number of Actors of the `RemoteDataPipe` type to
    which the data is being distributed during processing.

    Arguments:
        processors (list[BaseDataProcessor | DataPipe]):
            the initial pipe of processors
        num_proc (None | int):
            number of distributed workers to spawn. By default this
            value is taken from the `num_proc` argument to the
            `DistributedDataPipe.apply` function. However this can
            be set explicitly to allow different degrees of
            parallelism for different components of the data pipe.
        **kwargs:
            arguments forwarded to `ray.remote` function, specify
            the required resources here. For more infomation please
            refer to the ray documentation.
    """

    def __init__(
        self,
        processors: list[BaseDataProcessor, DataPipe] = [],
        num_proc: None | int = None,
        **kwargs,
    ) -> None:
        super(DistributedDataPipe, self).__init__(processors)

        self.num_proc = num_proc
        self._spawn_kwargs = kwargs

        self._actors = []
        self._idle_actor_ids: Queue = None

    def _spawn(self) -> ActorHandle:
        return (
            ray.remote(**self._spawn_kwargs)
            if len(self._spawn_kwargs) > 0
            else ray.remote
        )(RemoteDataPipe).remote(list(self))

    def _spawn_actors(self, num_actors: int) -> None:
        self._idle_actor_ids = Queue(maxsize=num_actors)
        # spawn actors for self
        for rank in range(num_actors):
            self._actors.append(self._spawn())
            self._idle_actor_ids.put(rank)

    def _kill_actors(self) -> None:
        # all actors should be idleing
        assert len(self._actors) == self._idle_actor_ids.qsize()
        # kill all actors and empty out the idle actor queue
        for actor in self._actors:
            ray.kill(actor)
            self._idle_actor_ids.get()
        # delete queue
        del self._idle_actor_ids
        self._idle_actor_ids = None

    def _check_actor(self, actor: ActorHandle) -> None:
        """Check configuration of the given actor"""
        assert self.is_prepared == ray.get(actor._self.remote("is_prepared"))

        if self.is_prepared:
            for feature_name in [
                "in_features",
                "new_features",
                "out_features",
            ]:
                target = getattr(self, feature_name)
                query = ray.get(actor._self.remote(feature_name))
                assert check_feature_equals(query, target)

    def prepare(self, features: datasets.Features) -> datasets.Features:
        """Prepare the distributed data pipe

        This includes the preparation of the main data pipe as well as
        all remote data pipes.

        Arguments:
            in_features (datasets.Features): input features

        Return:
            out_features (datasets.Features): output features
        """
        # prepare main process data pipe
        out_features = super(DistributedDataPipe, self).prepare(features)
        assert self.is_prepared

        # prepare all actors
        for actor_out_features in ray.get(
            [actor.prepare.remote(features) for actor in self._actors]
        ):
            assert check_feature_equals(actor_out_features, out_features)

        # check actors
        for actor in self._actors:
            self._check_actor(actor)

        return out_features

    def batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
        return_index: bool = False,
    ) -> dict[str, list[Any]]:
        """Process a batch of examples

        Sends the batch of examples to the a remote data pipe. The selection
        strategy of which remote data pipe to send the data to is implemented
        by the `_get_actor` function.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank
            return_index (bool):
                whether to return the source index for each output example

        Returns:
            out (dict[str, list[Any]]): processed examples
            idx (list[int]):
                the source index of each processed example, only returned
                when `return_index` is set
        """

        # select and actor and overwrite the rank
        rank = self._idle_actor_ids.get()
        actor = self._actors[rank]
        # call function on actor and get output
        output = ray.get(
            actor.batch_process.remote(
                examples=examples,
                index=index,
                rank=rank,
                return_index=return_index,
            )
        )
        # add the actor id back into the idle queue
        self._idle_actor_ids.put(rank)

        return output

    def iter_batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
        return_index: bool = False,
    ) -> Iterable[dict[str, list[Any]]]:
        raise NotImplementedError()

    def apply(
        self,
        data: (
            datasets.Dataset
            | datasets.DatasetDict
            | datasets.IterableDataset
            | datasets.IterableDatasetDict
        ),
        num_proc: None | int = None,
        **kwargs,
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Apply the data pipe to a dataset

        Arguments:
            data (Dataset|DatasetDict|IterableDataset|IterableDatasetDict):
                source dataset(s)
            **kwargs (dict[str, Any]):
                arguments forwarded to datasets `.map` function

        Returns:
            out (datasets.Dataset|datasets.DatasetDict): processed dataset(s)
        """

        if isinstance(
            data, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            raise NotImplementedError()

        # check num process argument
        if (
            (self.num_proc is not None)
            and (num_proc is not None)
            and (self.num_proc != num_proc)
        ):
            raise ValueError(
                "Ambiguous value for `num_proc`. Please specify the"
                "`num_proc` argument either as an argument to the"
                "constructor or the `apply` function of the"
                "`DistributedDataPipe`, but not both!"
            )

        num_proc = num_proc or self.num_proc or 1
        # spawn actors
        self._spawn_actors(num_actors=num_proc)
        # apply the distributed data pipe to the dataset
        # using no multiprocessing, the workload distribution to
        # different  workers is handled through ray in the
        # `batch_process` function
        data = super(DistributedDataPipe, self).apply(
            data, num_proc=None, **kwargs
        )
        # kill all actors
        self._kill_actors()

        return data
