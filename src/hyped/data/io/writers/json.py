import os
from types import SimpleNamespace
from typing import Any

import _io
import orjson

from .base import BaseDatasetWriter


class JsonDatasetWriter(BaseDatasetWriter):
    """Json Dataset Writer

    Implements the `BaseDatasetWriter` class to write a dataset
    to the disk in json-line format.

    Arguments:
        save_dir (str): the directory to save the dataset in
        exist_ok (bool):
            whether it is ok to write to the directory if it
            already exists. Defaults to False.
        num_proc (None | int):
            The number of processes to use. Defaults to the number of
            cpu cores available.
        tqdm_kwargs (dict[str, Any]):
            extra keyword arguments passed to the tqdm progress bar
        tqdm_update_interval (float):
            the update interval in seconds in which the tqdm bar
            is updated
    """

    def worker_shard_file_obj(
        self, path: str, worker_id: int
    ) -> _io.TextIOWrapper:
        return open(
            os.path.join(path, "data_shard_%i.jsonl" % worker_id), "wb+"
        )

    def finalize_worker(self, state: SimpleNamespace) -> None:
        """Cleanup and close the save file"""
        # finalize the worker
        super(JsonDatasetWriter, self).finalize_worker(state)

        # check if the worker ouput file exists after finalization
        # or if it got deleted because it was empty
        if os.path.isfile(state.save_file_path):
            # remove trailing newline character when the file
            # exists, i.e. contains content
            with open(state.save_file_path, "r+") as f:
                f.seek(f.seek(0, os.SEEK_END) - 1, os.SEEK_SET)
                f.truncate()

    def consume_example(
        self,
        shard_id: int,
        example_id: int,
        example: dict[str, Any],
        state: SimpleNamespace,
    ) -> None:
        """Encode an example to json and write it to the worker's save file.

        Arguments:
            worker (mp.Process): worker process
            worker_id (int): worker id
            shard_id (int): dataset shard id
            example_id (int): example id in the current dataset shard
            example (dict[str, Any]): the example to consume
            state (SimpleNamespace): worker state
        """
        # save example to file in json format
        state.save_file.write(orjson.dumps(example) + b"\n")
