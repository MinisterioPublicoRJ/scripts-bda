import re
import numpy as np

from copy import deepcopy
from unidecode import unidecode
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def clean_text(x):
    if not x:
        x = u''
    return re.sub('[^a-zA-Z ]', '', unidecode(x).upper())


class OneVsRestLogisticRegression:
    def __init__(self, negative_column_index=None, **kwargs):
        self.model_ = OneVsRestClassifier(LogisticRegression(**kwargs))
        self.negative_column_index_ = negative_column_index

    def fit(self, X, y):
        if self.negative_column_index_:
            self.model_.fit(
                X,
                np.delete(y, self.negative_column_index_, axis=1))
        else:
            self.model_.fit(
                X,
                y)

    def predict(self, X):
        p = self.model_.predict(X)
        if self.negative_column_index_:
            p = np.insert(
                p,
                self.negative_column_index_,
                values=(p.sum(axis=1) == 0).astype(int),
                axis=1)
        return p


class RegexClassifier:
    def __init__(self, rules):
        self.rules = deepcopy(rules)

    def predict(self, texts):
        results = []
        for t in texts:
            t_classes = []
            for c in self.rules:
                for expression in self.rules[c]:
                    m = re.search(expression, unidecode(t).upper())
                    if m:
                        t_classes.append(c)
                        break
            results.append(t_classes)
        return results
