# -*- coding: utf-8 -*-
import unittest
from utils import clean_text

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
    def test_predict(self):
        pass

class TestOneVsRestLogisticRegression(unittest.TestCase):
    def test_fit_with_negative_column(self):
        pass

    def test_fit_without_negative_column(self):
        pass

    def test_predict_with_negative_column(self):
        pass

    def test_predict_without_negative_column(self):
        pass