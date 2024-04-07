import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_key import FeatureKey
from tests.data.processors.base import BaseTestDataProcessor


class ConstantDataProcessorConfig(BaseDataProcessorConfig):
    """Configuration for `ConstantDataProcessor`

    Attributes:
        name (str): name of the feature to be added
        value (str): value of the feature to be added
    """

    name: str = "A"
    value: str = "B"


class ConstantDataProcessor(BaseDataProcessor[ConstantDataProcessorConfig]):
    """Data Processor that adds a constant string feature to every example"""

    def map_features(self, features):
        return Features({self.config.name: Value("string")})

    def process(self, example, *args, **kwargs):
        return {self.config.name: self.config.value}


class ConstantGeneratorDataProcessorConfig(ConstantDataProcessorConfig):
    """Configuration for `ConstantGeneratorDataProcessor`"""

    n: int = 3


class ConstantGeneratorDataProcessor(ConstantDataProcessor):
    """Data Processor that generates n examples from every source example
    and adds a constant string feature to each
    """

    # overwrite config type
    CONFIG_TYPE = ConstantGeneratorDataProcessorConfig

    def process(self, example, *args, **kwargs):
        for _ in range(self.config.n):
            yield super().process(example, *args, **kwargs)


class TestDataProcessorConfig(object):
    def test_extract_feature_keys(self):
        class Config(BaseDataProcessorConfig):
            # simple keys
            a: FeatureKey = "a"
            b: FeatureKey = "b"
            c: None | FeatureKey = None
            # list and dict of keys
            l: list[FeatureKey] = ["1", "2", "3"]  # noqa: E741
            ol: None | list[FeatureKey] = ["4", "5", "6"]
            d: dict[str, FeatureKey] = {"key1": "d1", "key2": "d2"}
            od: dict[str, FeatureKey] = {"key1": "d3", "key2": "d4"}
            # nested variations
            ll: list[list[FeatureKey]] = [["11", "12"], ["21", "22", "23"]]
            ld: list[dict[str, FeatureKey]] = [{"k1": "k1"}, {"k2": "k2"}]
            dl: dict[str, list[FeatureKey]] = {"k1": ["h", "i"], "k2": ["j"]}
            # no feature keys
            x: str = "x"
            y: None | int = 1
            z: tuple[str] = ("z",)

        assert set(list(Config().required_feature_keys)) == {
            FeatureKey("a"),
            FeatureKey("b"),
            FeatureKey("1"),
            FeatureKey("2"),
            FeatureKey("3"),
            FeatureKey("4"),
            FeatureKey("5"),
            FeatureKey("6"),
            FeatureKey("d1"),
            FeatureKey("d2"),
            FeatureKey("d3"),
            FeatureKey("d4"),
            FeatureKey("11"),
            FeatureKey("12"),
            FeatureKey("21"),
            FeatureKey("22"),
            FeatureKey("23"),
            FeatureKey("k1"),
            FeatureKey("k2"),
            FeatureKey("h"),
            FeatureKey("i"),
            FeatureKey("j"),
        }
        assert FeatureKey("c") in set(
            list(Config(c="c").required_feature_keys)
        )


class BaseTestSetup(BaseTestDataProcessor):
    @pytest.fixture(params=[True, False])
    def keep_inputs(self, request):
        return request.param

    @pytest.fixture
    def in_features(self):
        return Features({"X": Value("string")})

    @pytest.fixture
    def in_batch(self):
        return {"X": ["example %i" % i for i in range(10)]}


class TestDataProcessor(BaseTestSetup):
    @pytest.fixture
    def processor(self, keep_inputs):
        # create processor and make sure it is not prepared
        # before calling prepare function
        c = ConstantDataProcessorConfig(
            name="A", value="B", keep_input_features=keep_inputs
        )
        p = ConstantDataProcessor(c)
        assert not p.is_prepared
        # return processor
        return p

    @pytest.fixture
    def expected_out_features(self):
        return {"A": Value("string")}

    @pytest.fixture
    def expected_out_batch(self):
        return {"A": ["B"] * 10}


class TestDataProcessorWithOutputFormat(BaseTestSetup):
    @pytest.fixture
    def processor(self, keep_inputs):
        # create processor and make sure it is not prepared
        # before calling prepare function
        c = ConstantDataProcessorConfig(
            name="A",
            value="B",
            keep_input_features=keep_inputs,
            output_format={"custom_A": "A"},
        )
        print(c.output_format)
        p = ConstantDataProcessor(c)
        assert not p.is_prepared
        # return processor
        return p

    @pytest.fixture
    def expected_out_features(self):
        return {"custom_A": Value("string")}

    @pytest.fixture
    def expected_out_batch(self):
        return {"custom_A": ["B"] * 10}


class TestDataProcessorWithComplexOutputFormat(BaseTestSetup):
    @pytest.fixture
    def processor(self, keep_inputs):
        # create processor and make sure it is not prepared
        # before calling prepare function
        c = ConstantDataProcessorConfig(
            name="A",
            value="B",
            keep_input_features=keep_inputs,
            output_format={
                "custom_A": "A",
                "seq_A": ["A", "A"],
                "dict_A": {"A1": "A", "A2": "A"},
                "nest_A": [{"A3": {"A4": ["A"]}}],
            },
        )
        p = ConstantDataProcessor(c)
        assert not p.is_prepared
        # return processor
        return p

    @pytest.fixture
    def expected_out_features(self):
        return {
            "custom_A": Value("string"),
            "seq_A": Sequence(Value("string"), length=2),
            "dict_A": {"A1": Value("string"), "A2": Value("string")},
            "nest_A": Sequence(
                {"A3": {"A4": Sequence(Value("string"), length=1)}}, length=1
            ),
        }

    @pytest.fixture
    def expected_out_batch(self):
        return {
            "custom_A": ["B"] * 10,
            "seq_A": [["B", "B"]] * 10,
            "dict_A": [{"A1": "B", "A2": "B"}] * 10,
            "nest_A": [[{"A3": {"A4": ["B"]}}]] * 10,
        }


class TestGeneratorDataProcessor(BaseTestSetup):
    @pytest.fixture(params=[0, 1, 2, 3])
    def n(self, request):
        return request.param

    @pytest.fixture
    def processor(self, keep_inputs, n):
        # create processor and make sure it is not prepared
        # before calling prepare function
        c = ConstantGeneratorDataProcessorConfig(
            name="A", value="B", n=n, keep_input_features=keep_inputs
        )
        p = ConstantGeneratorDataProcessor(c)
        assert not p.is_prepared
        # return processor
        return p

    @pytest.fixture
    def expected_out_features(self):
        return {"A": Value("string")}

    @pytest.fixture
    def expected_out_batch(self, n):
        return {"A": ["B"] * 10 * n}
