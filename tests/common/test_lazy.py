import multiprocessing as mp
import os

import pytest

from hyped.common.lazy import LazyInstance, LazySharedInstance


def factory(pid=None):
    return "INSTANCE(%s)" % (pid or os.getpid())


class TestLazyInstance(object):
    @pytest.fixture
    def obj(self):
        return LazyInstance[str](factory)

    def test_case(self, obj):
        assert not obj._is_instantiated()
        # interact with the object and check the instance
        assert obj.lower() == factory().lower()
        assert obj._is_instantiated()


class TestLazySharedInstance(TestLazyInstance):
    def obj_factory(self):
        return LazySharedInstance[str]("test_shared_instance", factory)

    @pytest.fixture(scope="function")
    def obj(self):
        # save environment before creation of shared instance
        env = os.environ.copy()
        yield self.obj_factory()
        # reset the environment to the previous state
        os.environ.clear()
        os.environ.update(env)

    def _mp_worker(self, expected_val=None):
        # create expected value
        expected_val = expected_val or factory()

        obj = self.obj_factory()
        # interact with the object and check the instance
        assert obj.lower() == expected_val.lower()
        assert obj._is_instantiated()

    def test_case_mp(self, obj):
        # lazy shared instance is created but not instantiated
        assert not obj._is_instantiated()
        # interact with the object and check the instance
        assert obj.lower() == factory().lower()
        assert obj._is_instantiated()
        # expected value contains process id of parent process
        # and not the one of the child process which would be
        # the case when factory function would be called from
        # within the child process
        p = mp.Process(target=self._mp_worker, args=(factory(),))
        p.start()
        p.join()
        # check error in process
        assert p.exitcode == 0

    def test_case_mp_2(self, obj):
        # lazy shared instance is created but not instantiated
        assert not obj._is_instantiated()
        p = mp.Process(target=self._mp_worker)
        p.start()
        p.join()
        # check error in process
        assert p.exitcode == 0

        assert obj.lower() == factory(p.pid).lower()
