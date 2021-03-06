import mock
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from queries import (
    TRAIN_QUERY,
    PREDICT_QUERY,
    POSSIBLE_CLASSES_QUERY,
    SET_MODULE_QUERY,
    SET_CLIENT_QUERY,
    ATIV_SINDICANCIA_QUERY,
    DELETE_MOT_DECLARADO_QUERY,
    INSERT_MOT_DECLARADO_QUERY,
    get_train_data,
    get_predict_data,
    get_list_of_classes,
    set_module_and_client,
    get_max_dk,
    update_atividade_sindicancia,
    update_motivo_declarado
)


MOCK_DESCRIPTION = [['SNCA_DK'], ['DMDE_MDEC_DK']]
MOCK_COLUMNS = ['SNCA_DK', 'DMDE_MDEC_DK']
MOCK_VALUES = [[1, 2], [3, 4]]


@mock.patch('jaydebeapi.Cursor')
def test_get_train_data_without_UFED(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES

    expected_query = TRAIN_QUERY

    saida = get_train_data(_MockCursor, None)
    expected_output = pd.DataFrame(MOCK_VALUES,
                                   columns=MOCK_COLUMNS)

    assert_frame_equal(saida, expected_output)
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_get_train_data_with_UFED(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES

    expected_query = TRAIN_QUERY + " AND B.SNCA_UFED_DK = 33"

    saida = get_train_data(_MockCursor, 33)
    expected_output = pd.DataFrame(MOCK_VALUES,
                                   columns=MOCK_COLUMNS)

    assert_frame_equal(saida, expected_output)
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_get_train_data_UFED_not_int(_MockCursor):
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES
    with pytest.raises(TypeError):
        get_train_data(_MockCursor, 'not an int')


@mock.patch('jaydebeapi.Cursor')
def test_get_train_data_with_start_date(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES

    expected_query = TRAIN_QUERY + (" AND A.ATSD_DT_REGISTRO >= "
                                    "TO_DATE('2018-01-01', 'YYYY-MM-DD')")

    saida = get_train_data(_MockCursor, start_date='2018-01-01')
    expected_output = pd.DataFrame(MOCK_VALUES,
                                   columns=MOCK_COLUMNS)

    assert_frame_equal(saida, expected_output)
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_get_train_data_with_end_date(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES

    expected_query = TRAIN_QUERY + (" AND A.ATSD_DT_REGISTRO <= "
                                    "TO_DATE('2018-01-01', 'YYYY-MM-DD')")

    saida = get_train_data(_MockCursor, end_date='2018-01-01')
    expected_output = pd.DataFrame(MOCK_VALUES,
                                   columns=MOCK_COLUMNS)

    assert_frame_equal(saida, expected_output)
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_get_predict_data_without_UFED(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES

    expected_query = PREDICT_QUERY.format('', '')

    saida = get_predict_data(_MockCursor, None, only_null_class=False)
    expected_output = pd.DataFrame(MOCK_VALUES,
                                   columns=MOCK_COLUMNS)

    assert_frame_equal(saida, expected_output)
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_get_predict_data_with_UFED(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES

    expected_query = PREDICT_QUERY.format('', '') + " AND B.SNCA_UFED_DK = 33"

    saida = get_predict_data(_MockCursor, 33, only_null_class=False)
    expected_output = pd.DataFrame(MOCK_VALUES,
                                   columns=MOCK_COLUMNS)

    assert_frame_equal(saida, expected_output)
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_get_predict_data_UFED_not_int(_MockCursor):
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES
    with pytest.raises(TypeError):
        get_predict_data(_MockCursor, 'not an int')


@mock.patch('jaydebeapi.Cursor')
def test_get_predict_data_with_only_null(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES

    expected_query = PREDICT_QUERY.format('', '') + " AND D.DMDE_MDEC_DK = 13"

    saida = get_predict_data(_MockCursor, None)
    expected_output = pd.DataFrame(MOCK_VALUES,
                                   columns=MOCK_COLUMNS)

    assert_frame_equal(saida, expected_output)
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_get_predict_data_with_start_date(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES

    sd_string = (" AND A.ATSD_DT_REGISTRO >= "
                 "TO_DATE('2018-01-01', 'YYYY-MM-DD')")
    expected_query = PREDICT_QUERY.format(sd_string, '') + sd_string

    saida = get_predict_data(_MockCursor, start_date='2018-01-01',
                             only_null_class=False)
    expected_output = pd.DataFrame(MOCK_VALUES,
                                   columns=MOCK_COLUMNS)

    assert_frame_equal(saida, expected_output)
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_get_predict_data_with_end_date(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.description = MOCK_DESCRIPTION
    _MockCursor.fetchall.return_value = MOCK_VALUES

    ed_string = (" AND A.ATSD_DT_REGISTRO <= "
                 "TO_DATE('2018-01-01', 'YYYY-MM-DD')")
    expected_query = PREDICT_QUERY.format('', ed_string) + ed_string

    saida = get_predict_data(_MockCursor, end_date='2018-01-01',
                             only_null_class=False)
    expected_output = pd.DataFrame(MOCK_VALUES,
                                   columns=MOCK_COLUMNS)

    assert_frame_equal(saida, expected_output)
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_get_list_of_classes(_MockCursor):
    _MockCursor.execute.return_value = None
    _MockCursor.fetchall.return_value = [(1,), (2,)]

    expected_query = POSSIBLE_CLASSES_QUERY

    saida = get_list_of_classes(_MockCursor)
    expected_output = [1, 2]

    assert saida == expected_output
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_set_module_and_client(_MockCursor):
    expected_calls = [
        mock.call(SET_MODULE_QUERY),
        mock.call(SET_CLIENT_QUERY, ('DUNANT',))
    ]
    set_module_and_client(_MockCursor, 'DUNANT')
    _MockCursor.execute.assert_has_calls(expected_calls, any_order=True)


@mock.patch('jaydebeapi.Cursor')
def test_get_max_dk(_MockCursor):
    _MockCursor.fetchall.return_value = [[100]]

    expected_query = "SELECT MAX(NOME_DA_COLUNA) FROM NOME_DA_TABELA"

    saida = get_max_dk(_MockCursor, 'NOME_DA_TABELA', 'NOME_DA_COLUNA')
    expected_output = 100

    assert expected_output == saida
    _MockCursor.execute.assert_called_with(expected_query)


@mock.patch('jaydebeapi.Cursor')
def test_update_atividade(_MockCursor):
    update_atividade_sindicancia(_MockCursor, 1, 2, 'ROBO', 100)
    _MockCursor.execute.assert_called_with(ATIV_SINDICANCIA_QUERY,
                                           (1, 2, 'ROBO', 100))


@mock.patch('jaydebeapi.Cursor')
def test_update_motivo_declarado(_MockCursor):
    expected_calls = [
        mock.call(DELETE_MOT_DECLARADO_QUERY, (20,)),
        mock.call(INSERT_MOT_DECLARADO_QUERY, (20, 1)),
        mock.call(INSERT_MOT_DECLARADO_QUERY, (20, 2))
    ]
    update_motivo_declarado(_MockCursor, 20, [1, 2])
    _MockCursor.execute.assert_has_calls(expected_calls, any_order=True)
