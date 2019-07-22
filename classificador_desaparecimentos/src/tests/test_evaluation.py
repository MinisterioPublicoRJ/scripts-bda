import mock
import io
import ast
import pandas as pd

from pandas.util.testing import assert_frame_equal
from queries import (
    EVALUATE_QUERY,
    get_evaluate_data
)
from utils import (
    get_results_from_hdfs,
    expand_results,
    get_keys,
    get_percentage_of_change,
    get_number_of_modifications,
    generate_report
)


HDFS_RESULT_EXAMPLE = pd.DataFrame([
    {'SNCA_DK': 1, 'MDEC_DK': "(4.0,)"},
    {'SNCA_DK': 2, 'MDEC_DK': "(1.0, 2.0)"}
])

EXPANDED_RESULT_EXAMPLE = pd.DataFrame([
    {'SNCA_DK': 1, 'MDEC_DK': 4},
    {'SNCA_DK': 2, 'MDEC_DK': 1},
    {'SNCA_DK': 2, 'MDEC_DK': 2}
])

GET_KEYS_RESULT_EXAMPLE = [1, 2]

ORACLE_RESULT_EXAMPLE = pd.DataFrame([
    {'SNCA_DK': 1, 'MDEC_DK': 4},
    {'SNCA_DK': 2, 'MDEC_DK': 5}
])


@mock.patch('sklearn.preprocessing.MultiLabelBinarizer')
def test_generate_report(_MockBinarizer):
    _MockBinarizer.transform.side_effect = [
        [[0, 0, 0, 1, 0], [1, 1, 0, 0, 0]],
        [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    ]
    input_data = HDFS_RESULT_EXAMPLE.copy()
    input_data['MDEC_DK'] = input_data['MDEC_DK'].apply(
        lambda x: ast.literal_eval(x))
    output = generate_report(
        input_data,
        ORACLE_RESULT_EXAMPLE,
        _MockBinarizer)
    assert output.strip().startswith('precision')


@mock.patch('hdfs.InsecureClient')
def test_get_results_from_hdfs(_MockClient):
    _MockClient.list.return_value = ['1', '2']
    _MockClient.read.side_effect = [
        io.BytesIO(b'SNCA_DK,MDEC_DK\n1,"(4.0,)"'),
        io.BytesIO(b'SNCA_DK,MDEC_DK\n2,"(1.0, 2.0)"')]

    expected_output = HDFS_RESULT_EXAMPLE
    output = get_results_from_hdfs(_MockClient, 'path', 7)
    assert_frame_equal(expected_output, output, check_like=True)


def test_expand_results():
    expected_output = EXPANDED_RESULT_EXAMPLE
    input_data = HDFS_RESULT_EXAMPLE.copy()
    input_data['MDEC_DK'] = input_data['MDEC_DK'].apply(
        lambda x: ast.literal_eval(x))
    output = expand_results(input_data)
    assert_frame_equal(expected_output, output)


def test_get_keys():
    expected_output = GET_KEYS_RESULT_EXAMPLE
    output = get_keys(EXPANDED_RESULT_EXAMPLE, 'SNCA_DK')
    assert output == expected_output


@mock.patch('jaydebeapi.Cursor')
def test_get_evaluate_data(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = [['SNCA_DK'], ['MDEC_DK']]
    _MockCursor.fetchall.return_value = [[1, 4], [2, 5]]

    expected_query = EVALUATE_QUERY
    expected_output = ORACLE_RESULT_EXAMPLE
    output = get_evaluate_data(_MockCursor, GET_KEYS_RESULT_EXAMPLE)

    assert_frame_equal(expected_output, output, check_like=True)
    _MockCursor.execute.assert_called_with(expected_query)


def test_get_percentage_of_change():
    expected_output = 0.5
    output = get_percentage_of_change(
        EXPANDED_RESULT_EXAMPLE,
        ORACLE_RESULT_EXAMPLE)
    assert expected_output == output


def test_get_number_of_modifications():
    expected_output_rem = 1
    expected_output_add = 0
    expected_output_swap = 1
    output_rem, output_add, output_swap = get_number_of_modifications(
        EXPANDED_RESULT_EXAMPLE,
        ORACLE_RESULT_EXAMPLE)
    assert expected_output_rem == output_rem
    assert expected_output_add == output_add
    assert expected_output_swap == output_swap


# TODO: Think of a way to test metric saving function
@mock.patch('hdfs.InsecureClient')
def test_save_metrics_to_hdfs(_MockClient):
    pass
