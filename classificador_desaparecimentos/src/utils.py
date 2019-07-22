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


def get_results_from_hdfs(client, path, model_date=None):
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
    '''df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with
    each element moved into a new row. The values in the other columns are
    duplicated across the newly divided rows.
    '''
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
    return data[key_column].unique().tolist()


def get_percentage_of_change(data_hdfs, data_oracle):
    diffs = pd.concat([data_hdfs, data_oracle]).drop_duplicates(keep=False)
    nb_docs = len(data_oracle['SNCA_DK'].unique().tolist())
    nb_diffs = len(diffs['SNCA_DK'].unique().tolist())

    if nb_docs == 0:
        return 0.0
    else:
        return nb_diffs/float(nb_docs)


def get_number_of_modifications(data_hdfs, data_oracle):
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
    most_recent_model_date = sorted(client.list(path))[-1]
    for key, value in kwargs.items():
        client.write(
            '{}/{}/evaluate/{}.txt'.format(
                path,
                most_recent_model_date,
                key),
            str(value),
            overwrite=True)


def get_binarizer(client, path):
    most_recent_date = sorted(client.list(path))[-1]
    with client.read('{}/{}/model/mlb_binarizer.pkl'.format(
            path, most_recent_date)) as mlb_reader:
        mlb = pickle.loads(mlb_reader.read())
    return mlb


def generate_report(data_hdfs, data_oracle, binarizer):
    predictions = binarizer.transform(data_hdfs['MDEC_DK'])
    true_values = binarizer.transform(
        data_oracle.groupby('SNCA_DK')
                   .agg({lambda x: set(x)})
                   .reset_index()['MDEC_DK'])
    return classification_report(np.array(true_values), np.array(predictions))
