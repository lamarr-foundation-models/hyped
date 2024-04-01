import os
import datasets
from typing import Optional, Callable, TypeVar, Union, Literal, Dict

import ray
from ray.util.queue import Queue

from datasets.arrow_dataset import (
    transmit_tasks,
    transmit_format,
    NonExistentDatasetError,
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
from itertools import compress


@transmit_tasks
@transmit_format
def _map_dataset(
    self: datasets.Dataset,
    pipe: "DistributedDataPipe",
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
    """Adaption of the `datasets.Dataset.map` function tailored
    to applying a `DistributedDataPipe` instance to a dataset
    using ray`

    Arguments:
        self (datasets.Dataset): dataset
        pipe (DistributedDataPipe): data pipe
        batch_size (int):
            number of examples per batch provided to pipe.
            Defaults to 1000.
        drop_last_batch (bool):
            Whether a last batch smaller than the batch_size
            should be dropped instead of being processed
        keep_in_memory (bool):
            Keep the dataset in memory instead of writing it
            to a cache file.
        load_from_cache_file (bool):
            If a cache file storing the output of the pipe
            can be identified, use it instead of recomputing.
        cache_file_name (str):
            Provide the name of a path for the cache file.
        writer_batch_size (int):
            Number of rows per write operation for the cache
            file writer.
        disable_nullable (bool):
            Disallow null values in the table.
        new_fingerprint (str):
            The new fingerprint of the dataset after transform.
        desc (str):
            Meaningful description to be displayed alongside
            with the progress bar while mapping examples.

    Returns:
        transformed_dataset (datasets.Dataset):
            dataset after being passed through the data pipe

    """
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

    with pipe._pool.reserve_all() as actors:
        rank2actor = {a.rank: a for a in actors}
        rank2idx = {a.rank: i for i, a in enumerate(actors)}
        num_proc = num_shards = len(actors)

        if num_proc == 0:
            raise RuntimeError(
                "No remote actors available for "
                "`DistributedDataPipe` instance"
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

        # get cache file name
        if self.cache_files and (cache_file_name is None):
            cache_file_name = self._get_cache_file_path(new_fingerprint)

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
                num_shards=num_shards,
                index=index,
                contiguous=True,
                keep_in_memory=keep_in_memory,
            )
            for index in range(num_shards)
        ]

        futures = []
        update_queue = Queue()
        transformed_shards = [None] * num_shards
        # start all workers
        for actor, shard in zip(actors, shards):
            formatted_cache_file_name = format_cache_file_name(
                cache_file_name, actor.rank
            )
            formatted_new_fingerprint = format_new_fingerprint(
                new_fingerprint, actor.rank
            )

            try:
                idx = rank2idx[actor.rank]
                transformed_shards[idx] = load_processed_shard_from_cache(
                    dict(
                        shard=shard, cache_file_name=formatted_cache_file_name
                    )
                )
                # free actor
                actor.release()

            except NonExistentDatasetError:
                # start workload on actor
                futures.append(
                    actor.actor._map_single.remote(
                        shard=shard,
                        update_queue=update_queue,
                        rank=actor.rank,
                        offset=sum(map(len, shards[: actor.rank])),
                        cache_file_name=formatted_cache_file_name,
                        new_fingerprint=formatted_new_fingerprint,
                        **dataset_kwargs,
                    )
                )

        pbar_total = (
            len(self)
            if not drop_last_batch
            else (
                len(self) // num_shards // batch_size * num_shards * batch_size
            )
        )

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

                if isinstance(content, Exception):
                    raise RuntimeError(
                        "Error in remote worker with rank %i" % rank
                    ) from content

                if done:
                    shards_done += 1
                    idx = rank2idx[rank]
                    actor = rank2actor[rank]
                    # set content and release actor
                    transformed_shards[idx] = content
                    actor.release()
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

    # make sure all workers are finished
    ray.wait(futures, num_returns=len(futures), timeout=1)

    return result


def _map_dataset_dict(
    self: datasets.DatasetDict,
    pipe: "DistributedDataPipe",
    batch_size: Optional[int] = 1000,
    drop_last_batch: bool = False,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    cache_file_names: Optional[Dict[str, Optional[str]]] = None,
    writer_batch_size: Optional[int] = 1000,
    disable_nullable: bool = False,
    desc: Optional[str] = None,
) -> datasets.DatasetDict:
    """Adaption of the `datasets.DatasetDict.map` function tailored
    to applying a `DistributedDataPipe` instance to a dataset dict
    using ray`

    Arguments:
        self (datasets.DatasetDict): dataset dict
        pipe (DistributedDataPipe): data pipe
        batch_size (int):
            number of examples per batch provided to pipe.
            Defaults to 1000.
        drop_last_batch (bool):
            Whether a last batch smaller than the batch_size
            should be dropped instead of being processed
        keep_in_memory (bool):
            Keep the dataset in memory instead of writing it
            to a cache file.
        load_from_cache_file (bool):
            If a cache file storing the output of the pipe
            can be identified, use it instead of recomputing.
        cache_file_names (dict[str, Optional[str]]):
            Provide the name of a path for the cache file.
        writer_batch_size (int):
            Number of rows per write operation for the cache
            file writer.
        disable_nullable (bool):
            Disallow null values in the table.
        desc (str):
            Meaningful description to be displayed alongside
            with the progress bar while mapping examples.

    Returns:
        transformed_dataset (datasets.Dataset):
            dataset after being passed through the data pipe

    """

    self._check_values_type()
    if cache_file_names is None:
        cache_file_names = {k: None for k in self}

    return datasets.DatasetDict(
        {
            key: _map_dataset(
                self=data,
                pipe=pipe,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                keep_in_memory=keep_in_memory,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=cache_file_names[key],
                writer_batch_size=writer_batch_size,
                disable_nullable=disable_nullable,
                desc=desc,
            )
            for key, data in self.items()
        }
    )
