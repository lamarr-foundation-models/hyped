import os
import pickle
import tempfile
from time import sleep
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class LazyInstance(Generic[T]):
    """Lazy Instance

    Creates the object instance just as it is interacted with.

    Arguments:
        factory (Callable[[], T]): instance factory
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        self.factory = factory
        self.instance: None | T = None

    def __getattr__(self, name: str) -> Any:
        if self.instance is None:
            self.instance = self.factory()
        # forward all requests to the instance
        return getattr(self.instance, name)

    def _is_instantiated(self) -> bool:
        """Check whether the lazy object is instantiated"""
        return self.instance is not None


class LazySharedInstance(LazyInstance[T]):
    """Lazy Shared Instance

    Shares a lazy instance with all subprocesses

    The instance will be shared only with processes spawned after
    creating the lazy object, however the underlying instance can
    be created after.

    Arguments:
        identifier (str):
            instance identifier used to track instance accross processes
        factory (Callable[[], T]): instance factory
    """

    def __init__(self, identifier: str, factory: Callable[[], T]) -> None:
        # environment keys used to share the object
        env_key = "__HYPED_SHARED_INSTANCE_%s" % identifier

        # that is executed in the parent process only
        if env_key not in os.environ:
            # file name storing object
            # mark as registered but not instantiated yet
            tmp_file_name = tempfile.NamedTemporaryFile().name
            os.environ[env_key] = tmp_file_name
            # create registered instance file
            if not os.path.isfile("%s.registered" % tmp_file_name):
                open("%s.registered" % tmp_file_name, "w").close()

        else:
            tmp_file_name = os.environ[env_key]

        def wrapped_factory():
            if os.path.isfile("%s.registered" % tmp_file_name):
                # mark instance as pending
                os.rename(
                    "%s.registered" % tmp_file_name,
                    "%s.pending" % tmp_file_name,
                )
                # create instance
                instance = factory()
                # write instance to file
                with open("%s.pending" % tmp_file_name, "wb+") as f:
                    f.write(pickle.dumps(instance))
                # mark instance as ready to use
                os.rename("%s.pending" % tmp_file_name, tmp_file_name)
                # return the instance
                return instance

            else:
                # wait for the instance to be ready
                while os.path.isfile("%s.pending" % tmp_file_name):
                    sleep(0.1)
                assert os.path.isfile(tmp_file_name)
                # load instance from file
                with open(tmp_file_name, "rb") as f:
                    return pickle.loads(f.read())

        super(LazySharedInstance, self).__init__(wrapped_factory)
