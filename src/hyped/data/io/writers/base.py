import json
import os
from abc import ABC, abstractmethod
from types import SimpleNamespace

import _io
from torch.utils.data._utils.worker import get_worker_info

from hyped.utils.consumer import BaseDatasetConsumer


class BaseDatasetWriter(BaseDatasetConsumer, ABC):
    """Base Dataset Writer

    Implements the `BaseDatasetConsumer` class to write a dataset
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

    def __init__(
        self, save_dir: str, exist_ok: bool = False, **kwargs
    ) -> None:
        # initialize consumer
        super(BaseDatasetWriter, self).__init__(**kwargs)
        # create save directory if needed
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=exist_ok)

    @abstractmethod
    def worker_shard_file_obj(
        self, path: str, worker_id: int
    ) -> _io.TextIOWrapper:
        """Return the file object used to store the data consumed by the
        worker of a given id.

        Arguments:
            path (str): path to store the file in
            worker_id (int): worker id

        Returns:
            file (_io.TextIOWrapper): file used by the worker of the given id
        """
        ...

    def initialize_worker(self, state: SimpleNamespace) -> None:
        """Open the save file for the worker"""

        worker_info = get_worker_info()
        # open data save file
        state.save_file = self.worker_shard_file_obj(
            self.save_dir, worker_info.id if worker_info is not None else 1
        )
        # store file paths
        state.save_file_path = state.save_file.name
        state.features_file_path = os.path.join(self.save_dir, "features.json")

        if (worker_info is None) or (worker_info.id == 0):
            # save the datasets features
            with open(state.features_file_path, "w+") as f:
                f.write(json.dumps(state.dataset.features.to_dict()))

    def finalize_worker(self, state: SimpleNamespace) -> None:
        """Cleanup and close the save file"""

        # check if the file is empty
        state.save_file.seek(0, os.SEEK_END)
        is_empty = state.save_file.tell() == 0
        # close the file
        state.save_file.close()

        # delete the file if it is empty
        if is_empty:
            os.remove(state.save_file_path)
