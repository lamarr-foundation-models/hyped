from __future__ import annotations
import ray
import datasets
from ray.actor import ActorHandle
from ray.util.queue import Queue
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

    def _set_actors(
        self, i: int, idle_actor_ids: Queue, actors: list[ActorHandle]
    ) -> None:
        """Set actors and idle actor ids queue in a distributed data
        pipe at a specific index of the remote data pipe"""

        # make sure the processor at the given index
        # is a distributed data pipe
        if not isinstance(self[i], DistributedDataPipe):
            raise TypeError(
                "Expected `DistributedDataPipe` instance at index "
                "%i, got %s" % (i, self[i])
            )
        # set the queue and actors list
        self[i]._set_actors(idle_actor_ids, actors)


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
            `DistributedDataPipe.apply` function. However, this can
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

        self._spawn_kwargs = kwargs
        # keep track of all worker actors and which ones
        # are currently idleing
        self._actors = []
        self._idle_actor_ids: Queue = None
        # spawn all actors if number of processes is specified
        if num_proc is not None:
            self._spawn_actors(num_actors=num_proc)

    def _set_actors(
        self, idle_actor_ids: Queue, actors: list[ActorHandle]
    ) -> None:
        """Set the idle actor ids queue and actors list"""
        self._actors = actors
        self._idle_actor_ids = idle_actor_ids

    @property
    def are_actors_ready(self) -> bool:
        """Checks whether the worker actors are spawned"""
        return (self._idle_actor_ids is not None) and (len(self._actors) > 0)

    @property
    def num_proc(self) -> int | None:
        """Number of distributed workers/processes used.
        Returns None if the actors are not ready."""
        return len(self._actors) if self.are_actors_ready else None

    def _spawn_nested_actors(self) -> None:
        assert self.are_actors_ready

        for i, p in enumerate(self):
            if isinstance(p, DistributedDataPipe) and not p.are_actors_ready:
                # spawn actors for data pipe
                # note that this recursively calls the
                # _spawn_nested_actors function
                p._spawn_actors(num_actors=self.num_proc)
                # update the actors list and idle queue
                # in all spawned actors of the nested pipe
                for a in self._actors:
                    a._set_actors.remote(i, p._idle_actor_ids, p._actors)

            elif isinstance(p, DataPipe) and not isinstance(
                p, DistributedDataPipe
            ):
                # look for distributed data pipes
                # nested in standard data pipes
                for pp in p:
                    if isinstance(pp, DistributedDataPipe):
                        raise NotImplementedError()

    def _spawn_actors(self, num_actors: int) -> None:
        if self.are_actors_ready:
            raise RuntimeError(
                "Actors of `DistributedDataPipe` are already " "initialized"
            )

        self._idle_actor_ids = Queue(maxsize=num_actors)
        # remote worker spawn function
        spawn = lambda: (
            ray.remote(**self._spawn_kwargs)
            if len(self._spawn_kwargs) > 0
            else ray.remote
        )(RemoteDataPipe).remote(list(self))
        # spawn all actors for the current
        for rank in range(num_actors):
            self._actors.append(spawn())
            self._idle_actor_ids.put(rank)
        # spawn actors for nested distributed data pipes
        self._spawn_nested_actors()

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

        if not self.are_actors_ready:
            raise RuntimeError(
                "Actors of `DistributedDataPipe` not initialized. "
                "This occurs when a standard `DataPipe` instance "
                "contains a `DistributedDataPipe`."
            )

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

        # get the number of processes specified
        # in the arguments
        num_proc = kwargs.pop("num_proc", None)

        # check the number of processes argument
        if (
            self.are_actors_ready
            and (num_proc is not None)
            and (num_proc != self.num_proc)
        ):
            raise ValueError(
                "Got ambiguous values for `num_proc` argument. "
                "Please provide the argument either in the "
                "constructor or the `apply` function, but not both."
                "Got %i != %i" % (self.num_proc, num_proc)
            )

        elif not self.are_actors_ready:
            # spawn remote actors
            self._spawn_actors(
                num_actors=num_proc if num_proc is not None else 1
            )
            assert self.are_actors_ready

        return super(DistributedDataPipe, self).apply(data, **kwargs)
