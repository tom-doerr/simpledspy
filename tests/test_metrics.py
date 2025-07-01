"""Tests for metrics module"""

import pytest
from simpledspy.metrics import dict_exact_match_metric


class TestDictExactMatchMetric:
    """Test cases for dict_exact_match_metric function"""

    def test_empty_example_empty_prediction(self):
        """Test when both example and prediction are empty"""
        assert dict_exact_match_metric({}, {}) == 1.0

    def test_empty_example_non_empty_prediction(self):
        """Test when example is empty but prediction is not"""
        assert dict_exact_match_metric({}, {"key": "value"}) == 0.0

    def test_exact_match(self):
        """Test exact match between example and prediction"""
        example = {"a": 1, "b": 2, "c": 3}
        prediction = {"a": 1, "b": 2, "c": 3}
        assert dict_exact_match_metric(example, prediction) == 1.0

    def test_partial_match(self):
        """Test partial match between example and prediction"""
        example = {"a": 1, "b": 2, "c": 3}
        prediction = {"a": 1, "b": 2, "c": 4}
        assert dict_exact_match_metric(example, prediction) == 2 / 3

    def test_no_match(self):
        """Test no match between example and prediction"""
        example = {"a": 1, "b": 2}
        prediction = {"a": 3, "b": 4}
        assert dict_exact_match_metric(example, prediction) == 0.0

    def test_missing_keys_in_prediction(self):
        """Test when prediction is missing keys from example"""
        example = {"a": 1, "b": 2, "c": 3}
        prediction = {"a": 1, "b": 2}
        assert dict_exact_match_metric(example, prediction) == 2 / 3

    def test_extra_keys_in_prediction(self):
        """Test when prediction has extra keys not in example"""
        example = {"a": 1, "b": 2}
        prediction = {"a": 1, "b": 2, "c": 3}
        assert dict_exact_match_metric(example, prediction) == 1.0

    def test_tuple_prediction(self):
        """Test when prediction is a tuple"""
        example = {"output_0": "hello", "output_1": "world"}
        prediction = ("hello", "world")
        assert dict_exact_match_metric(example, prediction) == 1.0

    def test_tuple_prediction_partial_match(self):
        """Test tuple prediction with partial match"""
        example = {"output_0": "hello", "output_1": "world"}
        prediction = ("hello", "earth")
        assert dict_exact_match_metric(example, prediction) == 0.5

    def test_non_dict_non_tuple_prediction(self):
        """Test when prediction is neither dict nor tuple"""
        example = {"output": "hello"}
        prediction = "hello"
        assert dict_exact_match_metric(example, prediction) == 1.0

    def test_non_dict_non_tuple_prediction_no_match(self):
        """Test non-dict/non-tuple prediction with no match"""
        example = {"output": "hello"}
        prediction = "world"
        assert dict_exact_match_metric(example, prediction) == 0.0

    def test_with_none_values(self):
        """Test handling of None values"""
        example = {"a": None, "b": 2}
        prediction = {"a": None, "b": 2}
        assert dict_exact_match_metric(example, prediction) == 1.0

    def test_with_nested_structures(self):
        """Test with nested dictionaries"""
        example = {"a": {"x": 1}, "b": [1, 2, 3]}
        prediction = {"a": {"x": 1}, "b": [1, 2, 3]}
        assert dict_exact_match_metric(example, prediction) == 1.0

    def test_with_third_parameter(self):
        """Test that third parameter is ignored"""
        example = {"a": 1}
        prediction = {"a": 1}
        assert dict_exact_match_metric(example, prediction, "ignored") == 1.0

    def test_edge_case_single_key(self):
        """Test edge case with single key"""
        example = {"only_key": "value"}
        prediction = {"only_key": "value"}
        assert dict_exact_match_metric(example, prediction) == 1.0

    def test_edge_case_single_key_mismatch(self):
        """Test edge case with single key mismatch"""
        example = {"only_key": "value1"}
        prediction = {"only_key": "value2"}
        assert dict_exact_match_metric(example, prediction) == 0.0
