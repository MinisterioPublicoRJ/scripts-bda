import re

import numpy as np
from copy import deepcopy
from unidecode import unidecode
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class OneVsRestLogisticRegression:
    """Implements a Logistic Regression for a mutilabel problem
    using a OneVsRest approach.

    Attributes:
        model_: The model to be fit and used for prediction.
        negative_column_index_: The index of the column indicating the absence
            of other classes, if it exists.
    """

    def __init__(self, negative_column_index=None, **kwargs):
        """The constructor for the OneVsRestLogisticRegression class.

        Parameters:
            negative_column_index: The index of the column indicating the
                absence of other classes. None by default.
            kwargs: Arguments to be passed to the LogisticRegression class.
        """
        self.model_ = OneVsRestClassifier(LogisticRegression(**kwargs))
        self.negative_column_index_ = negative_column_index

    def fit(self, X, y):
        """Fits the model to the data and its labels.

        If negative_column_index_ is not None, then the corresponding column
        in y is removed before fitting the model.

        Parameters:
            X: Numpy array with the features of the data to fit the model to.
            y: Numpy array with the one hot encoding of the labels of the data
                to fit the model to.The labels must be in an n x m matrix,
                where n is the number of data points, and m the number of
                classes.
        """
        if self.negative_column_index_:
            self.model_.fit(
                X,
                np.delete(y, self.negative_column_index_, axis=1))
        else:
            self.model_.fit(
                X,
                y)

    def predict(self, X, threshold=0.7, max_classes=3):
        """Predicts the labels for a given dataset.

        If negative_column_index_ is not None, then the corresponding column
        is reinserted after prediction. The values of this column will be
        equal to 1 in the cases where no class has been predicted.

        Parameters:
            X: Numpy array with the data that will be used for prediction.

        Returns:
            p: Numpy array with the predictions.
        """
        prob = self.model_.predict_proba(X)
        consider = np.argsort(prob)[:, :-(max_classes+1):-1]
        mult = np.zeros(prob.shape)
        for a, b in zip(mult, consider):
            a[b] = 1
        prob = mult*prob

        p = (prob >= threshold).astype(int)

        if self.negative_column_index_:
            p = np.insert(
                p,
                self.negative_column_index_,
                values=(p.sum(axis=1) == 0).astype(int),
                axis=1)
        return p


class RegexClassifier:
    """Implements a classifier based on RegEx.

    Attributes:
        rules: A dictionary where each key corresponds to a class label, and
            the values are lists of expressions corresponding to that class.
    """

    def __init__(self, rules):
        """The constructor for the RegexClassifier.

        Parameters:
            rules: A dictionary where each key corresponds to a class label,
                and the values are lists of expressions corresponding to that
                class.
        """
        self.rules = deepcopy(rules)

    def predict(self, texts):
        """Predicts the labels for a given dataset.

        Parameters:
            texts: An iterable where each element is a string.

        Returns:
            results: The predictions made according to the regex rules.
        """
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