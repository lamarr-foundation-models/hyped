from contextlib import nullcontext

import pytest
from datasets import Features, Sequence, Value

from hyped.common.feature_checks import check_feature_equals
from hyped.common.feature_key import FeatureKey, FeatureKeyCollection


class TestFeatureKey(object):
    def test_basics(self):
        with pytest.raises(
            ValueError, match="First entry of a feature key must be a string"
        ):
            FeatureKey(1)

        # test basics on single entry key
        key = FeatureKey("key")
        assert isinstance(key, FeatureKey)
        assert len(key) == 1
        assert isinstance(key[0], str) and (key[0] == "key")

        # test basics multi-entry key
        key = FeatureKey("key", 1, slice(5))
        assert len(key) == 3
        assert isinstance(key[0], str)
        assert isinstance(key[1], int)
        assert isinstance(key[2], slice)
        # test slicing
        assert isinstance(key[:1], FeatureKey)
        assert isinstance(key[1:], tuple) and not isinstance(
            key[1:], FeatureKey
        )

    @pytest.mark.parametrize(
        "key,is_simple",
        [
            (FeatureKey("key"), True),
            (FeatureKey("key", "val"), True),
            (FeatureKey("key", slice(None)), True),
            (FeatureKey("key", "val", slice(None)), True),
            (FeatureKey("key", "val", slice(3)), False),
            (FeatureKey("key", 2, "val"), False),
        ],
    )
    def test_is_simple(self, key, is_simple):
        # make sure key is simple
        assert is_simple == key.is_simple
        # check raise condition
        with nullcontext() if is_simple else pytest.raises(
            TypeError, match="Expected simple key, got"
        ):
            key.raise_is_simple()

    @pytest.mark.parametrize(
        "key,cutoff_key",
        [
            (FeatureKey("key"), FeatureKey("key")),
            (FeatureKey("A", "B", 0, "C"), FeatureKey("A", "B", 0, "C")),
            (FeatureKey("key", slice(-1)), FeatureKey("key")),
            (
                FeatureKey("A", "B", 0, "C", slice(-1), "D"),
                FeatureKey("A", "B", 0, "C"),
            ),
            (
                FeatureKey("A", "B", slice(-1), 0, "C", slice(-1), "D"),
                FeatureKey("A", "B"),
            ),
        ],
    )
    def test_cutoff_at_slice(self, key, cutoff_key):
        assert key.cutoff_at_slice() == cutoff_key

    @pytest.mark.parametrize(
        "key,simple_key",
        [
            (FeatureKey("key"), FeatureKey("key")),
            (FeatureKey("A", "B", 0, "C"), FeatureKey("A", "B")),
            (FeatureKey("key", slice(None)), FeatureKey("key", slice(None))),
            (
                FeatureKey("A", "B", slice(None), 0, "C"),
                FeatureKey("A", "B", slice(None)),
            ),
        ],
    )
    def test_simple_subkey(self, key, simple_key):
        assert key.simple_subkey == simple_key

    @pytest.mark.parametrize(
        "key,features,feature",
        [
            (
                FeatureKey("key"),
                Features({"key": Value("int32")}),
                Value("int32"),
            ),
            (
                FeatureKey("A", "B"),
                Features({"A": {"B": Value("int32")}}),
                Value("int32"),
            ),
            (
                FeatureKey("A", 0),
                Features({"A": Sequence(Value("int32"))}),
                Value("int32"),
            ),
            (
                FeatureKey("A", 1),
                Features({"A": Sequence(Value("int32"))}),
                Value("int32"),
            ),
            (
                FeatureKey("A", slice(None)),
                Features({"A": Sequence(Value("int32"))}),
                Sequence(Value("int32")),
            ),
            (
                FeatureKey("A", slice(None)),
                Features({"A": Sequence(Value("int32"), length=10)}),
                Sequence(Value("int32"), length=10),
            ),
            (
                FeatureKey("A", slice(5)),
                Features({"A": Sequence(Value("int32"), length=10)}),
                Sequence(Value("int32"), length=5),
            ),
            (
                FeatureKey("A", slice(-3)),
                Features({"A": Sequence(Value("int32"), length=10)}),
                Sequence(Value("int32"), length=7),
            ),
            (
                FeatureKey("A", slice(2, 8, 2)),
                Features({"A": Sequence(Value("int32"), length=10)}),
                Sequence(Value("int32"), length=3),
            ),
        ],
    )
    def test_index_features(self, key, features, feature):
        assert check_feature_equals(key.index_features(features), feature)

    @pytest.mark.parametrize(
        "key,features,exc_type",
        [
            (FeatureKey("key"), Features({"X": Value("int32")}), KeyError),
            (
                FeatureKey("A", "B"),
                Features({"A": {"X": Value("int32")}}),
                KeyError,
            ),
            (
                FeatureKey("A", 1),
                Features({"A": Sequence(Value("int32"), length=1)}),
                IndexError,
            ),
        ],
    )
    def test_errors_on_index_features(self, key, features, exc_type):
        with pytest.raises(exc_type):
            key.index_features(features)

    @pytest.mark.parametrize(
        "key,example,value",
        [
            (FeatureKey("key"), {"key": 5}, 5),
            (FeatureKey("A", "B"), {"A": {"B": 5}}, 5),
            (
                FeatureKey("A", slice(None)),
                {"A": list(range(10))},
                list(range(10)),
            ),
            (
                FeatureKey("A", slice(5)),
                {"A": list(range(10))},
                list(range(5)),
            ),
            (
                FeatureKey("A", slice(3, 8)),
                {"A": list(range(10))},
                list(range(3, 8)),
            ),
            (
                FeatureKey("A", slice(3, 8, 2)),
                {"A": list(range(10))},
                list(range(3, 8, 2)),
            ),
        ],
    )
    def test_index_example(self, key, example, value):
        assert key.index_example(example) == value

    @pytest.mark.parametrize(
        "key,batch,values",
        [
            (FeatureKey("key"), {"key": [5]}, [5]),
            (FeatureKey("A", "B"), {"A": [{"B": 5}]}, [5]),
            (FeatureKey("A"), {"A": list(range(10))}, list(range(10))),
            (
                FeatureKey("A", "B"),
                {"A": [{"B": i} for i in range(10)]},
                list(range(10)),
            ),
            (
                FeatureKey("A", 3),
                {"A": [list(range(5)) for i in range(10)]},
                [3 for _ in range(10)],
            ),
            (
                FeatureKey("A", slice(2, 4)),
                {"A": [list(range(5)) for i in range(10)]},
                [list(range(2, 4)) for _ in range(10)],
            ),
        ],
    )
    def test_index_batch(self, key, batch, values):
        assert key.index_batch(batch) == values

    @pytest.mark.parametrize(
        "key,features,out",
        [
            (FeatureKey("key"), Features({"key": Value("int32")}), Features()),
            (
                FeatureKey("A"),
                Features({"A": Value("int32"), "B": Value("int32")}),
                Features({"B": Value("int32")}),
            ),
            (
                FeatureKey("A", "X"),
                Features({"A": {"X": Value("int32"), "Y": Value("int32")}}),
                Features({"A": {"Y": Value("int32")}}),
            ),
            (
                FeatureKey("A", 0),
                Features({"A": Sequence(Value("int32"))}),
                Features({"A": Sequence(Value("int32"))}),
            ),
            (
                FeatureKey("A", 0),
                Features({"A": Sequence(Value("int32"), length=2)}),
                Features({"A": Sequence(Value("int32"), length=1)}),
            ),
            (
                FeatureKey("A", slice(None), "X"),
                Features(
                    {
                        "A": Sequence(
                            {"X": Value("int32"), "Y": Value("int32")},
                            length=2,
                        )
                    }
                ),
                Features({"A": Sequence({"Y": Value("int32")}, length=2)}),
            ),
            (
                FeatureKey("A", slice(0, 2)),
                Features({"A": Sequence(Value("int32"), length=2)}),
                Features({"A": Sequence(Value("int32"), length=0)}),
            ),
        ],
    )
    def test_remove_from_features(self, key, features, out):
        assert check_feature_equals(key.remove_from_features(features), out)

    @pytest.mark.parametrize(
        "key,example,out",
        [
            (FeatureKey("key"), {"key": 1}, {}),
            (FeatureKey("A"), {"A": 0, "B": 1}, {"B": 1}),
            (FeatureKey("A", "X"), {"A": {"X": 0, "Y": 1}}, {"A": {"Y": 1}}),
            (
                FeatureKey("A", 0),
                {"A": list(range(10))},
                {"A": list(range(1, 10))},
            ),
            (
                FeatureKey("A", slice(2, 8, 2)),
                {"A": list(range(10))},
                {"A": [0, 1, 3, 5, 7, 8, 9]},
            ),
            (
                FeatureKey("A", slice(None), "X"),
                {"A": [{"X": 2 * i, "Y": i} for i in range(10)]},
                {"A": [{"Y": i} for i in range(10)]},
            ),
            (
                FeatureKey("A", slice(None), slice(None), "X"),
                {
                    "A": [
                        [{"X": i, "Y": j * i} for i in range(10)]
                        for j in range(3)
                    ]
                },
                {"A": [[{"Y": j * i} for i in range(10)] for j in range(3)]},
            ),
        ],
    )
    def test_remove_from_example(self, key, example, out):
        assert key.remove_from_example(example) == out

    @pytest.mark.parametrize(
        "key,batch,out",
        [
            (FeatureKey("key"), {"key": list(range(10))}, {}),
            (
                FeatureKey("A"),
                {"A": list(range(10)), "B": list(range(10))},
                {"B": list(range(10))},
            ),
            (
                FeatureKey("A", "X"),
                {"A": [{"X": i, "Y": i} for i in range(10)]},
                {"A": [{"Y": i} for i in range(10)]},
            ),
            (
                FeatureKey("A", 0),
                {"A": [list(range(10))]},
                {"A": [list(range(1, 10))]},
            ),
            (
                FeatureKey("A", slice(2, 8, 2)),
                {"A": [list(range(10))]},
                {"A": [[0, 1, 3, 5, 7, 8, 9]]},
            ),
            (
                FeatureKey("A", slice(None), "X"),
                {"A": [[{"X": 2 * i, "Y": i} for i in range(10)]]},
                {"A": [[{"Y": i} for i in range(10)]]},
            ),
            (
                FeatureKey("A", slice(None), slice(None), "X"),
                {
                    "A": [
                        [
                            [{"X": i, "Y": j * i} for i in range(10)]
                            for j in range(3)
                        ]
                    ]
                },
                {"A": [[[{"Y": j * i} for i in range(10)] for j in range(3)]]},
            ),
        ],
    )
    def test_remove_from_batch(self, key, batch, out):
        assert key.remove_from_batch(batch) == out

    @pytest.mark.parametrize(
        "features,expected_keys",
        [
            (Features({"key": Value("int32")}), [FeatureKey("key")]),
            (
                Features({"A": Value("int32"), "B": Value("int32")}),
                [FeatureKey("A"), FeatureKey("B")],
            ),
            (
                Features({"A": {"X": Value("int32"), "Y": Value("int32")}}),
                [FeatureKey("A", "X"), FeatureKey("A", "Y")],
            ),
            (
                Features({"A": Sequence(Value("int32"))}),
                [FeatureKey("A", slice(None))],
            ),
            (
                Features({"A": Sequence(Value("int32"), length=6)}),
                [FeatureKey("A", i) for i in range(6)],
            ),
        ],
    )
    def test_iter_keys_in_features(self, features, expected_keys):
        keys = list(FeatureKey.iter_keys_in_features(features))
        assert len(keys) == len(expected_keys)
        assert all(key in keys for key in expected_keys)


class TestFeatureKeyCollection(object):
    def test_basics(self):
        col = FeatureKeyCollection(
            {
                "a": FeatureKey("a"),
                "b": "b",
                "c": ("x", "y"),
                "d": [FeatureKey("a"), "b"],
                "e": {"x": "x", "y": FeatureKey("y")},
            }
        )

        assert isinstance(col["a"], FeatureKey)
        assert isinstance(col["b"], FeatureKey)
        assert isinstance(col["c"], FeatureKey)
        assert all(isinstance(k, FeatureKey) for k in col["d"])
        assert all(isinstance(k, FeatureKey) for k in col["e"].values())

        col["x"] = "x"
        col["y"] = ("x", "y")
        col["z"] = ["x", "y"]
        col["k"] = {"x": "x", "y": FeatureKey("y")}

        assert isinstance(col["x"], FeatureKey)
        assert isinstance(col["y"], FeatureKey)
        assert all(isinstance(k, FeatureKey) for k in col["z"])
        assert all(isinstance(k, FeatureKey) for k in col["k"].values())

    @pytest.mark.parametrize(
        "keys,expected_collection",
        [
            ([FeatureKey("a")], FeatureKeyCollection({"a": FeatureKey("a")})),
            (
                [FeatureKey("a"), FeatureKey("b")],
                FeatureKeyCollection(
                    {"a": FeatureKey("a"), "b": FeatureKey("b")}
                ),
            ),
            (
                [FeatureKey("a", "b", "c"), FeatureKey("a", "x")],
                FeatureKeyCollection(
                    {
                        "a": {
                            "b": {"c": FeatureKey("a", "b", "c")},
                            "x": FeatureKey("a", "x"),
                        },
                    }
                ),
            ),
        ],
    )
    def test_from_features(self, keys, expected_collection):
        assert (
            FeatureKeyCollection.from_feature_keys(keys) == expected_collection
        )

    @pytest.mark.parametrize(
        "collection,expected_keys",
        [
            (FeatureKeyCollection({"a": FeatureKey("a")}), [FeatureKey("a")]),
            (
                FeatureKeyCollection(
                    {"a": FeatureKey("a"), "b": FeatureKey("b")}
                ),
                [FeatureKey("a"), FeatureKey("b")],
            ),
            (
                FeatureKeyCollection(
                    {
                        "a": FeatureKey("a"),
                        "b": {str(i): FeatureKey("b", i) for i in range(5)},
                    }
                ),
                [FeatureKey("a")] + [FeatureKey("b", i) for i in range(5)],
            ),
            (
                FeatureKeyCollection(
                    {
                        "a": FeatureKey("a"),
                        "b": [FeatureKey("b", i) for i in range(5)],
                    }
                ),
                [FeatureKey("a")] + [FeatureKey("b", i) for i in range(5)],
            ),
        ],
    )
    def test_feature_keys(self, collection, expected_keys):
        print(collection)
        keys = list(collection.feature_keys)
        assert len(keys) == len(expected_keys)
        assert all(key in keys for key in expected_keys)

    @pytest.mark.parametrize(
        "collection,features,expected_features",
        [
            (
                FeatureKeyCollection({"a": FeatureKey("x")}),
                Features({"x": Value("int32")}),
                Features({"a": Value("int32")}),
            ),
            (
                FeatureKeyCollection({"a": {"b": FeatureKey("x")}}),
                Features({"x": Value("int32")}),
                Features({"a": {"b": Value("int32")}}),
            ),
            (
                FeatureKeyCollection(
                    {"a": [FeatureKey("x"), FeatureKey("y")]}
                ),
                Features({"x": Value("int32"), "y": Value("int32")}),
                Features({"a": Sequence(Value("int32"), length=2)}),
            ),
        ],
    )
    def test_collect_features(self, collection, features, expected_features):
        assert check_feature_equals(
            collection.collect_features(features), expected_features
        )

    @pytest.mark.parametrize(
        "collection,example,expected_values",
        [
            (FeatureKeyCollection({"a": FeatureKey("x")}), {"x": 5}, {"a": 5}),
            (
                FeatureKeyCollection({"a": {"b": FeatureKey("x")}}),
                {"x": 5},
                {"a": {"b": 5}},
            ),
            (
                FeatureKeyCollection(
                    {"a": [FeatureKey("x"), FeatureKey("y")]}
                ),
                {"x": 5, "y": 6},
                {"a": [5, 6]},
            ),
        ],
    )
    def test_collect_values(self, collection, example, expected_values):
        assert collection.collect_values(example) == expected_values

    @pytest.mark.parametrize(
        "collection,batch,expected_values",
        [
            (
                FeatureKeyCollection({"a": FeatureKey("x")}),
                {"x": list(range(10))},
                {"a": list(range(10))},
            ),
            (
                FeatureKeyCollection({"a": {"b": FeatureKey("x")}}),
                {"x": list(range(10))},
                {"a": [{"b": i} for i in range(10)]},
            ),
        ],
    )
    def test_collect_batch(self, collection, batch, expected_values):
        assert collection.collect_batch(batch) == expected_values
