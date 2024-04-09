from __future__ import annotations

import multiprocessing as mp
import multiprocessing.context
import multiprocessing.managers
import uuid
import warnings
from typing import Any, Iterable

from hyped.utils.lazy import LazyInstance, LazySharedInstance


class SyncManager(mp.managers.SyncManager):
    """Custom Sync Manager with special registered types"""


class StatisticsReportStorage(object):
    """Statistics Report Storage

    Internal class implementing a thread-safe storage for statistics
    reports. It manages the values and locks for statistics.

    Arguments:
        manager (mp.Manager):
            multiprocessing manager responsible for sharing statistic
            values and locks between processes
    """

    def __init__(self):
        global _manager
        # create shared storage for statistic values and locks
        self.stats = _manager.dict()
        self.locks = _manager.dict()
        # keep track of all registered keys
        self.registered_keys: set[str] = _manager.set()
        # create unique id for instance comparissons accross processes
        self._uuid = uuid.uuid4()

    @property
    def manager(self) -> SyncManager:
        """Multiprocessing manager instance"""
        global _manager
        return _manager

    def register(self, key: str, init_val: Any) -> None:
        """Register a statistic key to the storage

        Adds the initial value to the value storage and creates a lock
        dedicated to the statistic.

        Note that only the main process (i.e. the process that created the
        instance of the storage) can register new statistics to the storage.

        Arguments:
            key (str): statistic key under which to store the statistic
            init_val (Any): initial value of the statistic
        """
        # key already registered
        if key in self:
            raise RuntimeError(
                "Error while registering statistic key `%s`: "
                "Key already registered" % key
            )
        # create lock for the statistic
        lock = _manager.RLock()
        # write initial value and lock to dicts
        self.stats[key] = init_val
        self.locks[key] = lock
        # add key to registered keys
        self.registered_keys.add(key)

    def get_lock_for(self, key: str) -> mp.RLock:
        """Get the lock dedicated to a given statistic

        Arguments:
            key (str): statistic key

        Returns.
            lock (mp.RLock): multiprocessing lock dedicated to the statistic
        """
        # key not registered
        if key not in self:
            raise KeyError(
                "Error while getting lock for statistic: "
                "Statistic key `%s` not registered" % key
            )
        # get lock for key
        return self.locks[key]

    def get(self, key: str) -> Any:
        """Get the value of a given statistic

        Arguments:
            key (str): statistic key

        Returns:
            val (Any): statistic value
        """
        # key not registered
        if key not in self:
            raise KeyError(
                "Error while getting value for statistic: "
                "Statistic key `%s` not registered" % key
            )
        # get statistic value for key
        return self.stats[key]

    def set(self, key: str, val: Any) -> None:
        """Set the value of a given statistic

        Arguments:
            key (str): statistic key
            val (Any): value to store in the statistic
        """
        # key not registered
        if key not in self:
            raise KeyError(
                "Error while setting value for statistic: "
                "Statistic key `%s` not registered" % key
            )
        # update value in statistics dict
        with self.get_lock_for(key):
            self.stats[key] = val

    def __hash__(self):
        return hash(self._uuid)

    def __eq__(self, other):
        return isinstance(other, StatisticsReportStorage) and (
            self._uuid == other._uuid
        )

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, val: Any) -> None:
        self.set(key, val)

    def __contains__(self, key: str) -> bool:
        return key in self.registered_keys


class StatisticsReportManager(object):
    """Statistics Report Manager

    Internal class managing statistic report storages and the multiprocessing
    manager underlying the storages. It keeps track of the active reports,
    i.e. reports to which statistics should be written.
    """

    def __init__(self) -> None:
        # create set of active reports
        self._active_reports: set[StatisticsReportStorage] = self.manager.set()

    @property
    def manager(self) -> SyncManager:
        """Multiprocessing manager instance"""
        global _manager
        return _manager

    def is_empty(self) -> bool:
        """Boolean indicating whether there are any reports active"""
        return (self._active_reports is None) or (
            len(self._active_reports) == 0
        )

    def reports(self) -> Iterable[StatisticsReportStorage]:
        """Iterator over active report storages

        Warns when no reports are activated

        Returns:
            reports_iter (Iterable[StatisticsReportStorage]):
                iterator over active report storages
        """

        # warn when no reports are active
        if self.is_empty():
            warnings.warn(
                "No active statistic reports found. Computed statistics will "
                "not be tracked. Active a `StatisticsReport` instance to "
                "track statistics.",
                UserWarning,
            )
        # iterate over active reports
        return iter(self._active_reports)

    def new_statistics_report_storage(self) -> StatisticsReportStorage:
        """Create a new statistic report storage

        Returns:
            storage (StatisticReportStorage): new storage instance
        """
        return StatisticsReportStorage()

    def is_active(self, report: StatisticsReportStorage) -> bool:
        """Check if a given statistic report storage is active

        Arguments:
            storage (StatisticsReportStorage): storage instance to check for

        Returns:
            is_active (bool):
                boolean indicating whether the storage is active or not
        """
        return report in self._active_reports

    def activate(self, report: StatisticsReportStorage) -> None:
        """Activate a given statistics report storage in order for it to track
        computed statistics.

        Arguments:
            storage (StatisticsReportStorage): storage to activate
        """
        self._active_reports.add(report)

    def deactivate(self, report: StatisticsReportStorage) -> None:
        """Deactivate a given statistics report storage

        Arguments:
            storage (StatisticsReportStorage): storage to deactivate
        """
        if self.is_active(report):
            self._active_reports.remove(report)


class StatisticsReport(object):
    """Statistics Report

    Tracks statistics computed in data statistics processors. Activate the
    report to start tracking statistics.

    Can be used as a context manager.
    """

    def __init__(self) -> None:
        self.storage = (
            statistics_report_manager.new_statistics_report_storage()
        )

    @property
    def registered_keys(self) -> set[str]:
        """Set of registered statistic keys"""
        return set(self.storage.registered_keys)

    def get(self, key: str) -> Any:
        """Get the statistic value to a given key

        Arguments:
            key (str): statistic key

        Returns:
            val (Any): value of the statistic
        """
        return self.storage.get(key)

    def activate(self) -> None:
        """Activate the report"""
        statistics_report_manager.activate(self.storage)

    def deactivate(self) -> None:
        """Deactivate the report"""
        statistics_report_manager.deactivate(self.storage)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.registered_keys

    def __enter__(self) -> StatisticsReport:
        self.activate()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.deactivate()

    def __str__(self) -> str:
        return "\n".join(
            ["%s: %s" % (k, self.get(k)) for k in self.registered_keys]
        )


# register types to sync manager
SyncManager.register(
    "set",
    set,
    exposed=[
        "add",
        "clear",
        "copy",
        "difference",
        "difference_update",
        "discard",
        "intersection",
        "intersection_update",
        "isdisjoint",
        "issubset",
        "pop",
        "remove",
        "symmetric_difference",
        "symmetric_difference_update",
        "union",
        "update",
        "__or__",
        "__rand__",
        "__ror__",
        "__rsub__",
        "__rxor__",
        "__sub__",
        "__xor__",
        "__and__",
        "__eq__",
        "__iand__",
        "__ior__",
        "__isub__",
        "__ixor__",
        "__len__",
        "__contains__",
        "__iter__",
    ],
)
SyncManager.register(
    "StatisticsReportManager",
    StatisticsReportManager,
    exposed=[
        "is_empty",
        "reports",
        "new_statistics_report_storage",
        "is_active",
        "activate",
        "deactivate",
    ],
)


def _statistics_report_manager_factory() -> StatisticsReportManager:
    global _manager
    # create a shared instance of the statistics report manager
    return _manager.StatisticsReportManager()


def _sync_manager_factory() -> SyncManager:
    manager = SyncManager(ctx=mp.context.DefaultContext)
    manager.start()
    return manager


# create global variables as lazy instances
_manager: SyncManager = LazyInstance[SyncManager](_sync_manager_factory)
statistics_report_manager = LazySharedInstance[StatisticsReportManager](
    "statistics_report_manager", _statistics_report_manager_factory
)
