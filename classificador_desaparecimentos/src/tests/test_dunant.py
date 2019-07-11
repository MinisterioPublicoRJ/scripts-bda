# -*- coding: utf-8 -*-
import unittest
import numpy as np
import mock
from unittest import TestCase
from utils import clean_text, RegexClassifier, OneVsRestLogisticRegression
from sklearn.multiclass import OneVsRestClassifier


RULES = {
    1:  ['TESTE',
         'CHECAR'],
    2:  ['REAL',
         'PRODUCAO']
}

X_TEST = np.array([[1, 1.1], [0.1, 0.3], [0.9, 0.7]])
Y_TEST = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])


def test_db_connection():
    pass


def test_clean_text():
    test_input = u"AsdÁf23ça - éDLsão "
    expected_output = "ASDAFCA  EDLSAO "
    output = clean_text(test_input)

    assert output == expected_output


def test_clean_text_none():
    test_input = None
    expected_output = ""
    output = clean_text(test_input)

    assert output == expected_output


class TestRegexClassifier(unittest.TestCase):
    def test_predict_no_class(self):
        text_input = ['NADA', 'NOTHING']
        reg_clf = RegexClassifier(rules=RULES)
        output = reg_clf.predict(text_input)
        expected_output = [[], []]
        assert output == expected_output

    def test_predict_one_class(self):
        text_input = ['TESTE', 'NOTHING']
        reg_clf = RegexClassifier(rules=RULES)
        output = reg_clf.predict(text_input)
        expected_output = [[1], []]
        assert output == expected_output

    def test_predict_two_classes(self):
        text_input = ['TESTE REAL', 'PRODUCAO']
        reg_clf = RegexClassifier(rules=RULES)
        output = reg_clf.predict(text_input)
        expected_output = [[1, 2], [2]]
        assert output == expected_output


class TestOneVsRestLogisticRegression(unittest.TestCase):
    @mock.patch.object(OneVsRestClassifier, 'fit')
    def test_fit_with_negative_column(self, _mocked_fit):
        clf = OneVsRestLogisticRegression(negative_column_index=2)
        clf.fit(X_TEST, Y_TEST)

        _mocked_fit.assert_called_once()
        self.assertListEqual(_mocked_fit.call_args_list[0][0][0].tolist(), X_TEST.tolist())
        self.assertListEqual(_mocked_fit.call_args_list[0][0][1].tolist(), np.delete(Y_TEST, 2, axis=1).tolist())


    @mock.patch.object(OneVsRestClassifier, 'fit')
    def test_fit_without_negative_column(self, _mocked_fit):
        clf = OneVsRestLogisticRegression()
        clf.fit(X_TEST, Y_TEST)

        _mocked_fit.assert_called_once()
        self.assertListEqual(_mocked_fit.call_args_list[0][0][0].tolist(), X_TEST.tolist())
        self.assertListEqual(_mocked_fit.call_args_list[0][0][1].tolist(), Y_TEST.tolist())

    # Predict return result with right size
    @mock.patch.object(OneVsRestClassifier, 'predict')
    def test_predict_with_negative_column(self, _mocked_predict):
        _mocked_predict.return_value = np.delete(Y_TEST, 2, axis=1)
        clf = OneVsRestLogisticRegression(negative_column_index=2)
        saida = clf.predict(X_TEST)

        _mocked_predict.assert_called_once()
        self.assertListEqual(Y_TEST.tolist(), saida.tolist())


    @mock.patch.object(OneVsRestClassifier, 'predict')
    def test_predict_without_negative_column(self, _mocked_predict):
        _mocked_predict.return_value = Y_TEST
        clf = OneVsRestLogisticRegression()
        saida = clf.predict(X_TEST)

        _mocked_predict.assert_called_once()
        self.assertListEqual(Y_TEST.tolist(), saida.tolist())
