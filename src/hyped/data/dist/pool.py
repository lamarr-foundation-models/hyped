from __future__ import annotations
import ray
from ray.actor import ActorHandle
from ray.util.queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass
class _ReservedActor(object):
    """Helper Class"""

    _pool: ActorPool
    _rank: int
    _actor: ActorHandle

    _is_released: bool = field(init=False, default=False)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def actor(self) -> ActorHandle:
        return self._actor

    def __iter__(self):
        return iter([self.rank, self.actor])

    def release(self) -> None:
        if not self._is_released:
            self._pool._release(self.rank)
            self._is_released = True

    def __enter__(self) -> _ReservedActor:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.release()


class _ReservedActorList(list[_ReservedActor]):
    """Helper Class"""

    @property
    def ranks(self) -> list[int]:
        return [a.rank for a in self]

    @property
    def actors(self) -> list[ActorHandle]:
        return [a.actor for a in self]

    def for_all_actors(self, fn: Callable[[ActorHandle], T]) -> list[T]:
        return list(map(fn, self.actors))

    def release(self):
        for actor in self:
            actor.release()

    def __enter__(self) -> _ReservedActorList:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.release()


class ActorPool(object):
    """Ray Actor Pool

    Helper class managing a list of ray actors

    Arguments:
        actors (list[ActorHandle]): pool of ray actors
    """

    def __init__(self, actors: list[ActorHandle]) -> None:
        self.actors = actors
        self.idle_ranks = Queue(maxsize=len(actors))
        # add all ranks
        for rank in range(len(self.actors)):
            self.idle_ranks.put(rank)

    @property
    def num_actors(self) -> int:
        """Number of actors in the pool"""
        return len(self.actors)

    def reserve(self, **kwargs) -> _ReservedActor:
        """Reserve a single actor in the pool

        Arguments:
            block (bool):
                whether to block until an actor is idleing.
                Defaults to True.
            timeout (float):
                a timeout of how long to block before throwing
                a `ray.util.queue.Empty` exception.

        Returns:
            actor (_ReservedActor): reserved actor
        """
        # get idleing actor rank from queue and
        # the corresponsing actor
        rank = self.idle_ranks.get(**kwargs)
        actor = self.actors[rank]
        # return rank and actor
        return _ReservedActor(self, rank, actor)

    def reserve_all(self) -> _ReservedActorList:
        """Reserve all currently idleing actors in the pool

        Returns:
            actors (_ReservedActorList):
                list of all reserved actors
        """

        reserved_actors = _ReservedActorList()

        try:
            # collect all idleing actors
            while not self.idle_ranks.empty():
                actor = self.reserve(block=False)
                reserved_actors.append(actor)
        except Empty:
            pass

        return reserved_actors

    def _release(self, rank: int) -> None:
        """Internal function to release a reserved
        actor of a given rank


        Arguments:
            rank (int): rank of the actor to release
        """
        # put rank back to idle queue
        assert rank < self.num_actors
        self.idle_ranks.put(rank)

    def destroy(self) -> None:
        """Destroy all actors of the pool"""
        for actor in self.actors:
            ray.kill(actor)
