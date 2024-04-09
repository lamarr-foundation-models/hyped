import json
from copy import deepcopy
from typing import TypeVar

import pytest

from hyped.base.config import (
    AutoConfig,
    BaseAutoConfigurable,
    BaseConfig,
    BaseConfigurable,
)
from hyped.base.registry import default_registry

T = TypeVar("T")


class Configurable(BaseConfigurable[T]):
    """Non-abstract configurable type for easier testing"""

    def __init__(self, config):
        self.c = config

    @classmethod
    def from_config(cls, config):
        return cls(config)


@pytest.fixture(autouse=True)
def _reset_registry():
    # get registry state before test execution
    global_hash_register = default_registry.global_hash_register.copy()
    global_type_register = default_registry.global_type_register.copy()
    hash_tree = deepcopy(default_registry.hash_tree)

    # execute test
    yield

    # recover registry state
    default_registry.global_hash_register = global_hash_register
    default_registry.global_type_register = global_type_register
    default_registry.hash_tree = hash_tree


class TestBaseConfig:
    def test_dict_conversion(self):
        class A(BaseConfig):
            x: str = ""
            y: str = ""

        class B(A):
            z: str = ""

        a = A(x="x", y="y")
        b = B(x="a", y="b", z="c")
        # convert to dictionaties
        a_dict = a.to_dict()
        b_dict = b.to_dict()

        # test reconstruction from type hash
        assert a == AutoConfig.from_dict(a_dict)
        assert b == AutoConfig.from_dict(b_dict)

        # test reconstruction from type identifier
        a_dict.pop("__type_hash__")
        b_dict.pop("__type_hash__")
        assert a == AutoConfig.from_dict(a_dict)
        assert b == AutoConfig.from_dict(b_dict)

        # test reconstruction by explicit class
        a_dict.pop("type_id")
        b_dict.pop("type_id")
        assert a == A.from_dict(a_dict)
        assert b == B.from_dict(b_dict)

    def test_serialization(self):
        class A(BaseConfig):
            x: str = ""
            y: str = ""

        class B(A):
            z: str = ""

        a = A(x="x", y="y")
        b = B(x="a", y="b", z="c")
        # convert to dictionaties
        a_json = a.to_json()
        b_json = b.to_json()

        # test reconstruction from type hash
        assert a == AutoConfig.from_json(a_json)
        assert b == AutoConfig.from_json(b_json)

        # test reconstruction from type identifier
        a_dict = json.loads(a_json)
        b_dict = json.loads(b_json)
        a_dict.pop("__type_hash__")
        b_dict.pop("__type_hash__")
        a_json = json.dumps(a_dict)
        b_json = json.dumps(b_dict)
        assert a == AutoConfig.from_json(a_json)
        assert b == AutoConfig.from_json(b_json)

        # test reconstruction by explicit class
        a_dict.pop("type_id")
        b_dict.pop("type_id")
        a_json = json.dumps(a_dict)
        b_json = json.dumps(b_dict)
        assert a == A.from_json(a_json)
        assert b == B.from_json(b_json)


class TestBaseConfigurable:
    def test_config_type(self):
        class aConfig(BaseConfig):
            pass

        class bConfig(BaseConfig):
            pass

        class cConfig(bConfig):
            pass

        class dConfig(bConfig):
            pass

        class A(Configurable[aConfig]):
            pass

        class B(Configurable[bConfig]):
            pass

        class C(B):
            CONFIG_TYPE = cConfig

        class D(C):
            CONFIG_TYPE = dConfig

        # check config types
        assert A.config_type == aConfig
        assert B.config_type == bConfig
        assert C.config_type == cConfig
        assert D.config_type == dConfig
        # check type identifiers
        assert A.type_id.startswith(aConfig.type_id)
        assert B.type_id.startswith(bConfig.type_id)
        assert C.type_id.startswith(cConfig.type_id)
        assert D.type_id.startswith(dConfig.type_id)

    def test_config_type_error(self):
        class aConfig(BaseConfig):
            pass

        class bConfig(BaseConfig):
            pass

        class A(Configurable[aConfig]):
            pass

        # should raise type-error because config doesn't
        # inherit generic configuration set in supertype
        with pytest.raises(TypeError):

            class B(A):
                CONFIG_TYPE = bConfig

    def test_auto_from_config(self):
        class aConfig(BaseConfig):
            pass

        class bConfig(BaseConfig):
            pass

        class A(Configurable[aConfig]):
            pass

        class B(Configurable[bConfig]):
            pass

        class AutoConfigurable(BaseAutoConfigurable[Configurable]):
            pass

        class AutoA(BaseAutoConfigurable[A]):
            pass

        class AutoB(BaseAutoConfigurable[B]):
            pass

        # shared auto class
        assert isinstance(AutoConfigurable.from_config(aConfig()), A)
        assert isinstance(AutoConfigurable.from_config(bConfig()), B)
        # non-shared auto class
        assert isinstance(AutoA.from_config(aConfig()), A)
        assert isinstance(AutoB.from_config(bConfig()), B)
        # test target type out of scope
        with pytest.raises(ValueError):
            AutoB.from_config(aConfig())

        with pytest.raises(ValueError):
            AutoA.from_config(bConfig())
