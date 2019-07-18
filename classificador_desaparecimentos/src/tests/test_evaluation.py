import mock
import pandas as pd
import pytest

from pandas.util.testing import assert_frame_equal
from evaluate import (
    ORACLE_QUERY,
    get_results_from_hdfs,
    expand_results,
    get_keys,
    read_from_database,
    get_percentage_of_change,
    get_number_of_modifications,
    get_number_of_final_classes,
    save_metrics_to_hdfs
)


HDFS_RESULT_EXAMPLE = [
    {'SNCA_DK': 1, 'MDEC_DK': "(4.0,)"},
    {'SNCA_DK': 2, 'MDEC_DK': "(1.0, 2.0)"}
]

EXPANDED_RESULT_EXAMPLE = [
    {'SNCA_DK': 1, 'MDEC_DK': 4},
    {'SNCA_DK': 2, 'MDEC_DK': 1},
    {'SNCA_DK': 2, 'MDEC_DK': 2}
]

GET_KEYS_RESULT_EXAMPLE = [1, 2]

ORACLE_RESULT_EXAMPLE = [
    {'SNCA_DK': 1, 'MDEC_DK': 4},
    {'SNCA_DK': 2, 'MDEC_DK': 5}
]


@mock.patch('hdfs.InsecureClient')
def test_get_results_from_hdfs(_MockClient):
    expected_output = pd.DataFrame(HDFS_RESULT_EXAMPLE)
    output = get_results_from_hdfs(_MockClient, 7)
    assert_frame_equal(expected_output, output)


def test_expand_results():
    expected_output = pd.DataFrame(EXPANDED_RESULT_EXAMPLE)
    output = expand_results(HDFS_RESULT_EXAMPLE)
    assert_frame_equal(expected_output, output)


def test_get_keys():
    expected_output = GET_KEYS_RESULT_EXAMPLE
    output = get_keys(EXPANDED_RESULT_EXAMPLE, 'SNCA_DK')
    assert output == expected_output


@mock.patch('jaydebeapi.Cursor')
def test_read_from_database(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = [['SNCA_DK'], ['MDEC_DK']]
    _MockCursor.fetchall.return_value = [[1, 2], [4, 5]]

    expected_query = ORACLE_QUERY
    expected_output = pd.DataFrame(ORACLE_RESULT_EXAMPLE)
    output = read_from_database(_MockCursor, GET_KEYS_RESULT_EXAMPLE, 'SNCA_DK')

    assert_frame_equal(expected_output, output)
    _MockCursor.execute.assert_called_with(expected_query)


def test_get_percentage_of_change():
    expected_output = 0.5
    output = get_percentage_of_change(EXPANDED_RESULT_EXAMPLE, ORACLE_RESULT_EXAMPLE)
    assert expected_output == output


def test_get_number_of_modifications():
    expected_output = 2
    output = get_number_of_modifications(EXPANDED_RESULT_EXAMPLE, ORACLE_RESULT_EXAMPLE)
    assert expected_output == output


def test_get_number_of_final_classes():
    expected_output = 2
    output = get_number_of_final_classes(ORACLE_RESULT_EXAMPLE)
    assert expected_output == output


@mock.patch('hdfs.InsecureClient')
def test_save_metrics_to_hdfs(_MockClient):
    pass