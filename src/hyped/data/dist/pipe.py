from __future__ import annotations
import ray
import datasets
from ray.actor import ActorHandle
from ray.util.queue import Queue, Empty
from typing import Any, Iterable

from hyped.data.pipe import DataPipe
from hyped.data.processors.base import BaseDataProcessor
from hyped.utils.feature_checks import check_feature_equals


import os
from typing import Optional

from datasets.arrow_dataset import (
    transmit_tasks,
    transmit_format,
    _concatenate_map_style_datasets,
)
from datasets.fingerprint import (
    is_caching_enabled,
    update_fingerprint,
    validate_fingerprint,
    format_transform_for_fingerprint,
    format_kwargs_for_fingerprint,
)
from datasets.utils import tqdm as hf_tqdm


def _reserve_all_idle_actors(pipe: DistributedDataPipe) -> list[int]:
    # pipe actors not initialized
    if not pipe.are_actors_ready:
        return []
    # collect all idleing actor ids from the queue
    actor_ids = []
    try:
        while True:
            actor_ids.append(pipe._idle_actor_ids.get_nowait())
    except Empty:
        pass

    return actor_ids


def _free_actors(pipe: DistributedDataPipe, actor_ids: list[int]) -> None:
    # add all actors back to the idleing queue
    for i in actor_ids:
        pipe._idle_actor_ids.put(i)


@transmit_tasks
@transmit_format
def _map_dataset(
    self: datasets.Dataset,
    pipe: DistributedDataPipe,
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    drop_last_batch: bool = False,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    disable_nullable: bool = False,
    new_fingerprint: Optional[str] = None,
    suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
    desc: Optional[str] = None,
) -> datasets.Dataset:
    if keep_in_memory and cache_file_name is not None:
        raise ValueError(
            "Please use either `keep_in_memory` or "
            "`cache_file_name` but not both."
        )

    # If the array is empty we do nothing (but we make sure to
    # handle an empty indices mapping and remove the requested
    # columns anyway)
    if len(self) == 0:
        if self._indices is not None:  # empty indices mapping
            self = Dataset(
                self.data.slice(0, 0),
                info=self.info.copy(),
                split=self.split,
                fingerprint=new_fingerprint,
            )
        if remove_columns:
            return self.remove_columns(remove_columns)
        else:
            return self

    load_from_cache_file = (
        load_from_cache_file
        if load_from_cache_file is not None
        else is_caching_enabled()
    )

    # get all the idleing actors of the data pipe to use
    actor_ids = _reserve_all_idle_actors(pipe)
    actors = [pipe._actors[i] for i in actor_ids]
    # number of processes/actors/shards to use
    num_proc = num_shards = len(actor_ids)

    if num_proc == 0:
        raise RuntimeError(
            "No remote actors available for `DistributedDataPipe` " "instance"
        )

    dataset_kwargs = dict(
        batch_size=batch_size,
        drop_last_batch=drop_last_batch,
        keep_in_memory=keep_in_memory,
        writer_batch_size=writer_batch_size,
        disable_nullable=disable_nullable,
    )

    if new_fingerprint is None:
        # we create a unique hash from the function,
        # current dataset file and the mapping args
        transform = format_transform_for_fingerprint(
            datasets.Dataset._map_single
        )
        kwargs_for_fingerprint = format_kwargs_for_fingerprint(
            datasets.Dataset._map_single,
            (),
            dataset_kwargs | dict(shard=self, function=pipe),
        )
        kwargs_for_fingerprint["fingerprint_name"] = "new_fingerprint"
        new_fingerprint = update_fingerprint(
            self._fingerprint, transform, kwargs_for_fingerprint
        )
    else:
        validate_fingerprint(new_fingerprint)

    def load_processed_shard_from_cache(shard_kwargs):
        """Load a processed shard from cache if it exists,
        otherwise throw an error."""
        shard = shard_kwargs["shard"]
        # Check if we've already cached this computation (indexed by a hash)
        if shard_kwargs["cache_file_name"] is not None:
            if load_from_cache_file and os.path.exists(
                shard_kwargs["cache_file_name"]
            ):
                info = shard.info.copy()
                info.features = features
                info.task_templates = None
                return Dataset.from_file(
                    shard_kwargs["cache_file_name"],
                    info=info,
                    split=shard.split,
                )
        raise NonExistentDatasetError

    def format_cache_file_name(
        cache_file_name: Optional[str],
        rank: Union[int, Literal["*"]],  # noqa: F722
    ) -> Optional[str]:
        if not cache_file_name:
            return cache_file_name
        sep = cache_file_name.rindex(".")
        base_name, extension = cache_file_name[:sep], cache_file_name[sep:]
        if isinstance(rank, int):
            cache_file_name = (
                base_name
                + suffix_template.format(rank=rank, num_proc=num_proc)
                + extension
            )
            logger.info(f"Process #{rank} will write at {cache_file_name}")
        else:
            cache_file_name = (
                base_name
                + suffix_template.replace("{rank:05d}", "{rank}").format(
                    rank=rank, num_proc=num_proc
                )
                + extension
            )
        return cache_file_name

    def format_new_fingerprint(new_fingerprint: str, rank: int) -> str:
        new_fingerprint = new_fingerprint + suffix_template.format(
            rank=rank, num_proc=num_proc
        )
        validate_fingerprint(new_fingerprint)
        return new_fingerprint

    # create one dataset shard for each worker
    shards = [
        self.shard(
            num_shards=num_proc,
            index=rank,
            contiguous=True,
            keep_in_memory=keep_in_memory,
        )
        for rank in range(num_shards)
    ]

    pbar_total = (
        len(self)
        if not drop_last_batch
        else (len(self) // num_shards // batch_size * num_shards * batch_size)
    )

    # TODO: try to load shards from cache

    futures = []
    update_queue = Queue()
    # start all workers
    for rank, actor, shard in zip(actor_ids, actors, shards):
        futures.append(
            actor._map_single.remote(
                shard=shard,
                update_queue=update_queue,
                rank=rank,
                offset=sum(map(len, shards[:rank])),
                **dataset_kwargs,
                cache_file_name=format_cache_file_name(cache_file_name, rank),
                new_fingerprint=format_new_fingerprint(new_fingerprint, rank),
            )
        )

    transformed_shards = [None] * num_shards

    with hf_tqdm(
        unit=" examples",
        total=pbar_total,
        desc=(desc or "Map") + f" (num_proc={num_proc})",
    ) as pbar:
        # collect all outputs
        shards_done = 0
        while shards_done < num_shards:
            # get next value from update queue
            rank, done, content = update_queue.get()

            if done:
                shards_done += 1
                transformed_shards[rank] = content
            else:
                pbar.update(content)

    # concatenate all shards
    result = _concatenate_map_style_datasets(transformed_shards)

    # update fingerprint if the dataset changed
    if any(
        transformed_shard._fingerprint != shard._fingerprint
        for transformed_shard, shard in zip(transformed_shards, shards)
    ):
        result._fingerprint = new_fingerprint
    else:
        result._fingerprint = self._fingerprint

    # free actors
    ray.wait(futures, num_returns=len(futures))
    _free_actors(pipe, actor_ids)

    return result


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

    def _map_single(
        self, shard: datasets.Dataset, update_queue: Queue, **kwargs
    ) -> None:
        kwargs["batched"] = True
        kwargs["features"] = self.out_features
        kwargs["with_indices"] = True
        kwargs["with_rank"] = True

        for content in datasets.Dataset._map_single(
            shard=shard, function=self._batch_process_to_pyarrow, **kwargs
        ):
            update_queue.put(content)


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
