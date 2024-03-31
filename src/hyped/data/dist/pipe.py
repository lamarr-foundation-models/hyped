from __future__ import annotations
import ray
import datasets
from ray.actor import ActorHandle
from ray.util.queue import Queue, Empty
from typing import Any, Iterable

from hyped.data.pipe import DataPipe
from hyped.data.processors.base import BaseDataProcessor
from hyped.utils.feature_checks import check_feature_equals

from .pool import ActorPool
from .map import _map_dataset


class RemoteDataPipe(DataPipe):
    """(Internal) Remote Data Pipe

    Class that is distributed internally by `DistributedDataPipe`.
    Provides helper functions for distributed setting on top of
    the standard `DataPipe` functionality.
    """

    def _self(self, attr_name: str) -> Any:
        return getattr(self, attr_name)

    def _set_pool(self, idx: int | tuple[int], pool: ActorPool) -> None:
        pipe = self._at_idx(idx)
        # make sure the processor at the given index
        # is a distributed data pipe
        if not isinstance(pipe, DistributedDataPipe):
            raise TypeError(
                "Expected `DistributedDataPipe` instance at index "
                "%s, got %s" % (str(idx), pipe)
            )
        pipe._set_pool(pool)

    def _at_idx(self, idx: int | tuple[int]) -> ActorPool:
        idx = (idx,) if isinstance(idx, int) else idx

        p = self

        for j, i in enumerate(idx):
            # make sure the processor at the given index
            # is a distributed data pipe
            if not isinstance(p, DataPipe):
                raise TypeError(
                    "Expected `DataPipe` instance at index "
                    "%s, got %s" % (str(idx[:j]), p)
                )
            p = p[i]

        return p

    def _map_single(
        self, shard: datasets.Dataset, update_queue: Queue, **kwargs
    ) -> None:
        kwargs["batched"] = True
        kwargs["features"] = self.out_features
        kwargs["with_indices"] = True
        kwargs["with_rank"] = True

        try:
            for content in datasets.Dataset._map_single(
                shard=shard, function=self._batch_process_to_pyarrow, **kwargs
            ):
                update_queue.put(content)
        except Exception as e:
            update_queue.put((kwargs["rank"], False, e))


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
        # pool of all worker actors
        self._pool: None | ActorPool = None
        # spawn all actors if number of processes is specified
        if num_proc is not None:
            self._spawn_pool(num_actors=num_proc)

    def _set_pool(self, pool: ActorPool) -> None:
        assert not self.is_pool_ready
        self._pool = pool

    def _spawn_pool(self, num_actors: int) -> ActorPool:
        """Spawn all remote actors of the distributed data pipe
        including nested distributed data pipes.

        Arguments:
            num_actors (int): number of actors to spawn

        Returns:
            pool (ActorPool): actor pool of distributed data pipe
        """

        nested_pools = {}
        # spawn pool in nested data pipes
        for i, p in enumerate(self):
            if isinstance(p, DistributedDataPipe) and not p.is_pool_ready:
                # spawn the actor pool of the nested data pipe
                # use as many actors as the parent data pipe
                nested_pools[i] = p._spawn_pool(num_actors=num_actors)

            elif isinstance(p, DataPipe) and not isinstance(
                p, DistributedDataPipe
            ):
                # look for distributed data pipes
                # nested in standard data pipes
                if any(isinstance(x, DistributedDataPipe) for x in p):
                    raise NotImplementedError()

        # remote worker spawn function
        spawn = lambda: (
            ray.remote(**self._spawn_kwargs)
            if len(self._spawn_kwargs) > 0
            else ray.remote
        )(RemoteDataPipe).remote(list(self))
        # set actor pool for distributed data pipe
        self._set_pool(ActorPool([spawn() for _ in range(num_actors)]))

        # reserve all actors in the pool
        with self._pool.reserve_all() as reserved_actors:
            # set the actor pools of nested pipes in remote actors
            for i, pool in nested_pools.items():
                ray.wait(
                    reserved_actors.for_all_actors(
                        lambda a: a._set_pool.remote(i, p._pool)
                    ),
                    num_returns=len(reserved_actors),
                )

        return self._pool

    @property
    def is_pool_ready(self) -> bool:
        """Checks whether the actor pool is ready"""
        return self._pool is not None

    @property
    def num_proc(self) -> int | None:
        """Number of distributed workers/processes used.
        Returns None if the actors are not ready."""
        return self._pool.num_actors if self.is_pool_ready else None

    @property
    def is_prepared(self) -> bool:
        with self._pool.reserve_all() as actors:
            # TODO: make sure that the full pool is reserved and
            #       not just the actors that are currently idleing

            return (
                super(DistributedDataPipe, self).is_prepared
                and self.is_pool_ready
                and all(
                    ray.get(
                        actors.for_all_actors(
                            lambda a: a._self.remote("is_prepared")
                        )
                    )
                )
            )

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
        assert super(DistributedDataPipe, self).is_prepared

        if not self.is_pool_ready:
            raise RuntimeError(
                "Actor pool not initialized. Please make sure the "
                "pool is ready before calling `prepare`."
            )

        with self._pool.reserve_all() as reserved_actors:
            # make sure all actors are reserved for preparation
            # TODO: prepare is called recursively from within actors
            #       such that this check doesn't work for nested
            #       distributed data pipes
            # assert len(reserved_actors) == self.num_proc

            # prepare all actors
            for actor_out_features in ray.get(
                reserved_actors.for_all_actors(
                    lambda a: a.prepare.remote(features)
                )
            ):
                assert check_feature_equals(actor_out_features, out_features)

            # check actors
            for actor in reserved_actors.actors:
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

        if not self.is_pool_ready:
            raise RuntimeError(
                "Actors of `DistributedDataPipe` not initialized. "
                "This occurs when a standard `DataPipe` instance "
                "contains a `DistributedDataPipe`."
            )

        with self._pool.reserve() as (rank, actor):
            # call function on actor and get output
            output = ray.get(
                actor.batch_process.remote(
                    examples=examples,
                    index=index,
                    rank=rank,
                    return_index=return_index,
                )
            )

        return output

    def iter_batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
        return_index: bool = False,
    ) -> Iterable[dict[str, list[Any]]]:
        raise NotImplementedError()

    def _map(
        self,
        data: (
            datasets.Dataset
            | datasets.DatasetDict
            | datasets.IterableDataset
            | datasets.IterableDatasetDict
        ),
        **kwargs,
    ) -> (
        datasets.Dataset
        | datasets.DatasetDict
        | datasets.IterableDataset
        | datasets.IterableDatasetDict
    ):
        if isinstance(
            data,
            (
                datasets.DatasetDict,
                datasets.IterableDataset,
                datasets.IterableDatasetDict,
            ),
        ):
            raise NotImplementedError()

        if isinstance(data, datasets.Dataset):
            return _map_dataset(self=data, pipe=self, **kwargs)

    def apply(
        self,
        data: (
            datasets.Dataset
            | datasets.DatasetDict
            | datasets.IterableDataset
            | datasets.IterableDatasetDict
        ),
        **kwargs,
    ) -> (
        datasets.Dataset
        | datasets.DatasetDict
        | datasets.IterableDataset
        | datasets.IterableDatasetDict
    ):
        """Apply the data pipe to a dataset

        Arguments:
            data (Dataset|DatasetDict|IterableDataset|IterableDatasetDict):
                source dataset(s)
            **kwargs (dict[str, Any]):
                arguments forwarded to the `map` function used
                for the specific dataset type

        Returns:
            out (datasets.Dataset|datasets.DatasetDict): processed dataset(s)
        """

        # get the number of processes specified
        # in the arguments
        num_proc = kwargs.pop("num_proc", None)

        # destroy the pool after exection
        destroy_pool_after = not self.is_pool_ready

        # check the number of processes argument
        if (
            self.is_pool_ready
            and (num_proc is not None)
            and (num_proc != self.num_proc)
        ):
            raise ValueError(
                "Got ambiguous values for `num_proc` argument. "
                "Please provide the argument either in the "
                "constructor or the `apply` function, but not both."
                "Got %i != %i" % (self.num_proc, num_proc)
            )

        elif not self.is_pool_ready:
            # spawn remote actors
            self._spawn_pool(
                num_actors=num_proc if num_proc is not None else 1
            )
            assert self.is_pool_ready

        return super(DistributedDataPipe, self).apply(data, **kwargs)
