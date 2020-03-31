import re
import argparse

import numpy as np
import pandas as pd
from unidecode import unidecode
from sklearn.metrics import classification_report


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dunant Arguments")
    parser.add_argument('-os','--oracleServer', metavar='oracleServer', type=str, help='')
    parser.add_argument('-ou','--oracleUser', metavar='oracleUser', type=str, help='')
    parser.add_argument('-op','--oraclePassword', metavar='oraclePassword', type=str, help='')
    parser.add_argument('-od','--oracleDriverPath', metavar='oracleDriverPath', type=str, help='')

    parser.add_argument('-rnm','--robotName', metavar='robotName', type=str, help='')
    parser.add_argument('-rnb','--robotNumber', metavar='robotNumber', type=str, help='')

    parser.add_argument('-hs','--hdfsUrl', metavar='hdfsUrl', type=str, help='')
    parser.add_argument('-hu','--hdfsUser', metavar='hdfsUser', type=str, help='')
    parser.add_argument('-hd','--hdfsModelDir', metavar='hdfsModelDir', type=str, help='')

    parser.add_argument('-ut','--updateTables', dest='updateTables', action='store_true')
    parser.set_defaults(updateTables=False)

    parser.add_argument('-sdt','--startDate', metavar='startDate', type=str, default='', help='')
    parser.add_argument('-edt','--endDate', metavar='endDate', type=str, default='', help='')

    parser.add_argument('-no','--onlyNull', dest='onlyNull', action='store_true')
    parser.set_defaults(onlyNull=False)

    parser.add_argument('-th','--threshold', metavar='threshold', type=float, default=0.7, help='')
    parser.add_argument('-mc','--maxClasses', metavar='maxClasses', type=int, default=3, help='')

    parser.add_argument('-gs','--evaluateSaveGSpread', dest='evaluateSaveGSpread', action='store_true')
    parser.set_defaults(evaluateSaveGSpread=False)
    
    args = parser.parse_args()

    options = {
                    'oracle_server': args.oracleServer, 
                    'oracle_user': args.oracleUser,
                    'oracle_password' : args.oraclePassword,
                    'oracle_driver_path' : args.oracleDriverPath,
                    'robot_name' : args.robotName,
                    'robot_number': args.robotNumber,
                    'hdfs_url': args.hdfsUrl,
                    'hdfs_user': args.hdfsUser,
                    'hdfs_model_dir': args.hdfsModelDir,
                    'update_tables': args.updateTables,
                    'start_date': args.startDate,
                    'end_date': args.endDate,
                    'only_null': args.onlyNull,
                    'threshold': args.threshold,
                    'max_classes': args.maxClasses,
                    'evaluate_save_gspread': args.evaluateSaveGSpread
                }

    return options


def clean_text(x):
    """Transforms accented characters into unaccented characters
    and removes numbers and other symbols from a string.
    """
    if not x:
        x = u''
    return re.sub('[^a-zA-Z ]', '', unidecode(x).upper())


def get_results_from_hdfs(client, path, model_date=None):
    """Gets the predictions made by a model saved on HDFS.

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
        result_date = '{}/{}/{}'.format(r[6:8], r[4:6], r[:4])
        with client.read('{}/{}/results/{}'.format(
                path,
                model_date,
                r)) as results_reader:
            r_df = pd.read_table(results_reader, sep=',')
            r_df['DT_ACAO'] = result_date
            results.append(r_df)

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

    # Need to keep track of where each row is from after aggregation
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


def save_metrics_to_hdfs(client, path, model_name=None, **kwargs):
    """Save a set of metrics to HDFS on a given path.

    Parameters:
        client: The HDFS client to use.
        path: The path on which the model structure is.
        kwargs: The keys and values corresponding to the metrics to save. The
            keys are the metric names and the values are their values.
    """
    if not model_name:
        model_name = sorted(client.list(path))[-1]
    for key, value in kwargs.items():
        client.write(
            '{}/{}/evaluate/{}.txt'.format(
                path,
                model_name,
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
