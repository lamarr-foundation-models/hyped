from copy import deepcopy

import pytest

from hyped.base.registry import RegisterTypes, default_registry


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


class TestTypeRegistry:
    def test_registers(self):
        types = set(RegisterTypes.type_registry.types)
        type_ids = set(RegisterTypes.type_registry.type_ids)

        class A(RegisterTypes):
            pass

        # check simple case
        assert {A} == set(RegisterTypes.type_registry.types) - types
        assert {A.type_id} == set(
            RegisterTypes.type_registry.type_ids
        ) - type_ids

        # set up complex case
        class B(RegisterTypes):
            pass

        class C(B):
            pass

        class D(C, B):
            pass

        # check complex case
        assert {A, B, C, D} == set(RegisterTypes.type_registry.types) - types
        assert {A.type_id, B.type_id, C.type_id, D.type_id} == set(
            RegisterTypes.type_registry.type_ids
        ) - type_ids

        # test overwriting registered type ids
        class C(D):
            pass

        # should have a new type but the type id is overwritten
        assert {A, B, C, D, C} == set(
            RegisterTypes.type_registry.types
        ) - types
        assert {A.type_id, B.type_id, C.type_id, D.type_id} == set(
            RegisterTypes.type_registry.type_ids
        ) - type_ids

    def test_subtype_registers(self):
        types = set(RegisterTypes.type_registry.types)
        type_ids = set(RegisterTypes.type_registry.type_ids)

        class A(RegisterTypes):
            pass

        class B(A):
            pass

        class C(A):
            pass

        class D(B):
            pass

        class D2(C):
            pass

        # check types
        assert {A, B, C, D, D2} == set(
            RegisterTypes.type_registry.types
        ) - types
        assert {A, B, C, D, D2} == set(A.type_registry.types)
        assert {B, D} == set(B.type_registry.types)
        assert {C, D2} == set(C.type_registry.types)
        # check type ids
        assert {A.type_id, B.type_id, C.type_id, D.type_id, D2.type_id} == set(
            RegisterTypes.type_registry.type_ids
        ) - type_ids
        assert {A.type_id, B.type_id, C.type_id, D.type_id, D2.type_id} == set(
            A.type_registry.type_ids
        )
        assert {B.type_id, D.type_id} == set(B.type_registry.type_ids)
        assert {C.type_id, D2.type_id} == set(C.type_registry.type_ids)

    def test_get_type_by_hash(self):
        class A(RegisterTypes):
            pass

        class B(RegisterTypes):
            pass

        class C(B):
            pass

        assert A == RegisterTypes.type_registry.get_type_by_hash(A.type_hash)
        assert B == RegisterTypes.type_registry.get_type_by_hash(B.type_hash)
        assert C == RegisterTypes.type_registry.get_type_by_hash(C.type_hash)

    def test_get_type_by_t(self):
        class A(RegisterTypes):
            pass

        class B(RegisterTypes):
            pass

        class C(B):
            pass

        assert A == RegisterTypes.type_registry.get_type_by_t(A.type_id)
        assert B == RegisterTypes.type_registry.get_type_by_t(B.type_id)
        assert C == RegisterTypes.type_registry.get_type_by_t(C.type_id)
