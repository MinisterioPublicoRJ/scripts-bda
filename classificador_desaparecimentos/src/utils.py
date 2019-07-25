import re
import pickle

import numpy as np
import pandas as pd
from copy import deepcopy
from unidecode import unidecode
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report


def clean_text(x):
    """Transforms accented characters into unaccented characters
    and removes numbers and other symbols from a string.
    """
    if not x:
        x = u''
    return re.sub('[^a-zA-Z ]', '', unidecode(x).upper())


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

    def predict(self, X):
        """Predicts the labels for a given dataset.

        If negative_column_index_ is not None, then the corresponding column
        is reinserted after prediction. The values of this column will be
        equal to 1 in the cases where no class has been predicted.

        Parameters:
            X: Numpy array with the data that will be used for prediction.

        Returns:
            p: Numpy array with the predictions.
        """
        p = self.model_.predict(X)
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


def get_results_from_hdfs(client, path, model_date=None):
    """Gets the predictions made by a model saved on HDFS.

    The models are saved in the given file structure:
        path/
            /model_date1
                /model
                /results
                /evaluate
            /model_date2
                /model
                /results
                /evaluate
    Where /model_date is the date on which the model has been trained,
    using a YearMonthDayHourMinuteSecond notation.

    The /model directory contains the model components, such as the model
    itself, the multilabel binarizer, and vectorizer used, as well as the
    documents keys that were used for training.

    The /results directory contains the results given by the 'predict' phase,
    in a .csv format.

    The /evaluate directory holds the metrics calculated by the 'evaluate'
    phase

    Parameters:
        client: The HDFS Client to use to retrieve the predictions.
        path: The path on HDFS where the model is saved. This path should
            be relative to the home directory of the given Client.
        model_date: The date of the model to use, as per the
            YearMonthDayHourMinuteSecond notation.

    Returns:
        results: A Pandas DataFrame containing the results.
    """
    results = []

    # Get most recent model if no model_date is given
    if not model_date:
        model_date = sorted(client.list(path))[-1]

    result_list = client.list('{}/{}/results'.format(
        path,
        model_date))
    for r in result_list:
        with client.read('{}/{}/results/{}'.format(
                path,
                model_date,
                r)) as results_reader:
            results.append(pd.read_table(results_reader, sep=','))

    return pd.concat(results, ignore_index=True)


def expand_results(df, target_column='MDEC_DK'):
    """Splits a column which contains iterables into separate rows.

    This function will take a Pandas DataFrame and a column which contains
    iterables (for example, a column containing a list of ints), and split
    the list into several rows. A sort of inverse aggregate function.

    For instance, a row where target_column contains [1, 2] will become two
    rows containing, respectively, 1 and 2.

    Parameters:
        df: Pandas DataFrame to split.
        target_column: The column containing the values to split.

    Returns:
        A Pandas DataFrame with each entry for the target column separated,
        with each element moved into a new row. The values in the other
        columns are duplicated across the newly divided rows.
    """
    def splitListToRows(row, row_accumulator, target_column):
        split_row = row[target_column]
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = int(s)
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows, axis=1, args=(new_rows, target_column))
    new_df = pd.DataFrame(new_rows)
    return new_df


def get_keys(data, key_column):
    """Gets the unique keys present in the key_column of the data.

    Parameters:
        data: A Pandas DataFrame containing the data.
        key_column: The name of the column from which to get the keys.

    Returns:
        Python list containing the unique keys.
    """
    return data[key_column].unique().tolist()


def get_percentage_of_change(data_hdfs, data_oracle):
    """Get the percentage of documents that were altered after prediction.

    Parameters:
        data_hdfs: The predictions made by the model, coming from HDFS.
        data_oracle: The final classifications, coming from Oracle.

    Returns:
        A float from [0.0, 1.0] corresponding to the percentage of changed
        documents.
    """
    diffs = pd.concat([data_hdfs, data_oracle]).drop_duplicates(keep=False)
    nb_docs = len(data_oracle['SNCA_DK'].unique().tolist())
    nb_diffs = len(diffs['SNCA_DK'].unique().tolist())

    if nb_docs == 0:
        return 0.0
    else:
        return nb_diffs/float(nb_docs)


def get_number_of_modifications(data_hdfs, data_oracle):
    """The modifications made to the predictions to get the final classes.

    This function will represent the modifications made in three different
    sub-metrics: number of removals, number of additions, and number of swaps.
    The number of removals represent the number of predictions that had to be
    removed from the original predictions to get the final classifications. On
    the other hand, the number of additions give how many new additions had to
    be made. Lastly, the number of swaps give the number of modifications that
    had to be made to existing predictions (without adding or removing, just
    changing in place).

    Parameters:
        data_hdfs: The predictions made by the model, coming from HDFS.
        data_oracle: The final classifications, coming from Oracle.

    Returns:
        n_remove: Number of removals made from the predictions.
        n_add: Number of additions made to the predictions.
        n_swaps: Number of swaps made to the predictions.
    """
    hdfs_ind = data_hdfs.copy()
    oracle_ind = data_oracle.copy()

    hdfs_ind['FROM'] = 'HDFS'
    oracle_ind['FROM'] = 'ORACLE'

    diffs = pd.concat([hdfs_ind, oracle_ind])\
              .drop_duplicates(subset=['SNCA_DK', 'MDEC_DK'], keep=False)
    diffs = diffs.groupby(['SNCA_DK', 'FROM'])\
                 .count()\
                 .reset_index()\
                 .pivot('SNCA_DK', columns='FROM', values='MDEC_DK')

    n_remove = (diffs['HDFS'] - diffs['ORACLE']).apply(
        lambda x: x if x >= 0 else 0).sum()
    n_add = (diffs['ORACLE'] - diffs['HDFS']).apply(
        lambda x: x if x >= 0 else 0).sum()
    n_swaps = diffs.min(axis=1).sum()

    return n_remove, n_add, n_swaps


def save_metrics_to_hdfs(client, path, **kwargs):
    """Save a set of metrics to HDFS on a given path.

    Parameters:
        client: The HDFS client to use.
        path: The path on which the model structure is.
        kwargs: The keys and values corresponding to the metrics to save. The
            keys are the metric names and the values are their values.
    """
    most_recent_model_date = sorted(client.list(path))[-1]
    for key, value in kwargs.items():
        client.write(
            '{}/{}/evaluate/{}.txt'.format(
                path,
                most_recent_model_date,
                key),
            str(value),
            overwrite=True)


def generate_report(data_hdfs, data_oracle, binarizer):
    """Generate a classification report for the predictions.

    Parameters:
        data_hdfs: The predictions made by the model, coming from HDFS.
        data_oracle: The final classifications, coming from Oracle.
        binarizer: The MultiLabel Binarizer to one-hot encode the labels.

    Returns:
        A classification report for the given predictions and true values.
    """
    predictions = binarizer.transform(data_hdfs['MDEC_DK'])
    true_values = binarizer.transform(
        data_oracle.groupby('SNCA_DK')
                   .agg({lambda x: set(x)})
                   .reset_index()['MDEC_DK'])
    return classification_report(np.array(true_values), np.array(predictions))
