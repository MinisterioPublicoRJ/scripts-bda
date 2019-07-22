import ast
import sys
import pickle
import numpy as np
import pandas as pd
import jaydebeapi as jdbc
from decouple import config
from hdfs import InsecureClient
from sklearn.metrics import classification_report

URL_ORACLE_SERVER = config('URL_ORACLE_SERVER')
USER_ORACLE = config('USER_ORACLE')
PASSWD_ORACLE = config('PASSWD_ORACLE')
ORACLE_DRIVER_PATH = config('ORACLE_DRIVER_PATH')
ROBOT_NAME = config('ROBOT_NAME')
ROBOT_NUMBER = config('ROBOT_NUMBER')
HDFS_URL = config('HDFS_URL')
HDFS_USER = config('HDFS_USER')
HDFS_MODEL_DIR = config('HDFS_MODEL_DIR')
FORMATTED_HDFS_PATH = "/".join(HDFS_MODEL_DIR.split('/')[5:])

EVALUATE_QUERY = """
    SELECT DISTINCT A.DMDE_SDES_DK AS SNCA_DK, A.DMDE_MDEC_DK AS MDEC_DK
    FROM SILD.SILD_DESAPARE_MOT_DECLARADO A
    INNER JOIN SILD.SILD_ATIVIDADE_SINDICANCIA D
        ON A.DMDE_SDES_DK = D.ATSD_SNCA_DK
    WHERE D.ATSD_TPSN_DK = 22
"""

def get_results_from_hdfs(client, path, month):
    results = []

    most_recent_model_date = sorted(client.list(FORMATTED_HDFS_PATH))[-1]
    result_list = client.list('{}/{}/results'.format(
        FORMATTED_HDFS_PATH, 
        most_recent_model_date))

    for r in result_list:
        with client.read('{}/{}/results/{}'.format(
                FORMATTED_HDFS_PATH, 
                most_recent_model_date,
                r)) as results_reader:
            results.append(pd.read_table(results_reader, sep=','))

    return pd.concat(results, ignore_index=True)


def expand_results(df, target_column='MDEC_DK'):
    '''df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row,row_accumulator,target_column):
        split_row = row[target_column]
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = int(s)
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column))
    new_df = pd.DataFrame(new_rows)
    return new_df


def get_keys(data, key_column):
    return data[key_column].unique().tolist()


def read_from_database(cursor, keys):
    cursor.execute(EVALUATE_QUERY)

    columns = [desc[0] for desc in cursor.description]
    result = pd.DataFrame(cursor.fetchall(), columns=columns)

    return result[result['SNCA_DK'].isin(keys)]


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

    diffs = pd.concat([hdfs_ind, oracle_ind]).drop_duplicates(subset=['SNCA_DK', 'MDEC_DK'], keep=False)
    diffs = diffs.groupby(['SNCA_DK', 'FROM']).count().reset_index().pivot('SNCA_DK', columns='FROM', values='MDEC_DK')

    n_remove = (diffs['HDFS'] - diffs['ORACLE']).apply(lambda x: x if x >= 0 else 0).sum()
    n_add = (diffs['ORACLE'] - diffs['HDFS']).apply(lambda x: x if x >= 0 else 0).sum()
    n_swaps = diffs.min(axis=1).sum()

    return n_remove, n_add, n_swaps


def save_metrics_to_hdfs(client, **kwargs):
    most_recent_model_date = sorted(client.list(FORMATTED_HDFS_PATH))[-1]
    for key, value in kwargs.items():
        client.write(
        '{}/{}/evaluate/{}.txt'.format(
            FORMATTED_HDFS_PATH, 
            most_recent_model_date,
            key),
        str(value),
        overwrite=True)


def get_binarizer(client):
    most_recent_date = sorted(client.list(FORMATTED_HDFS_PATH))[-1]
    with client.read('{}/{}/model/mlb_binarizer.pkl'.format(
            FORMATTED_HDFS_PATH, most_recent_date)) as mlb_reader:
        mlb = pickle.loads(mlb_reader.read())
    return mlb


def generate_report(data_hdfs, data_oracle, binarizer):
    predictions = binarizer.transform(data_hdfs['MDEC_DK'])
    true_values = binarizer.transform(
        data_oracle.groupby('SNCA_DK')\
                   .agg({lambda x: set(x)})\
                   .reset_index()['MDEC_DK'])
    return classification_report(np.array(true_values), np.array(predictions))

print('Running Evaluate script:')

print('Connecting to HDFS and Oracle database...')
client = InsecureClient(HDFS_URL, HDFS_USER)

conn = jdbc.connect("oracle.jdbc.driver.OracleDriver",
                    URL_ORACLE_SERVER,
                    [USER_ORACLE, PASSWD_ORACLE],
                    ORACLE_DRIVER_PATH)
curs = conn.cursor()

mlb = get_binarizer(client)

print('Retrieving prediction results from HDFS...')
data_hdfs = get_results_from_hdfs(client, FORMATTED_HDFS_PATH, 1)
data_hdfs['MDEC_DK'] = data_hdfs['MDEC_DK'].apply(lambda x: ast.literal_eval(x))

keys = get_keys(data_hdfs, 'SNCA_DK')

print('Retrieving data from Oracle...')
data_oracle = read_from_database(curs, keys)
if len(data_oracle) == 0:
    sys.exit('No documents have been validated!')

print('Generating classification report...')
report = generate_report(data_hdfs, data_oracle, mlb)

print('Preparing HDFS data for next metrics...')
data_hdfs = expand_results(data_hdfs)
data_hdfs = data_hdfs[data_hdfs['SNCA_DK'].isin(data_oracle['SNCA_DK'].unique().tolist())]

print('Calculating evaluation metrics...')
nb_documents_predicted = len(keys)
nb_documents_validated = len(get_keys(data_oracle, 'SNCA_DK'))

pctg_change = get_percentage_of_change(data_hdfs, data_oracle)

nb_remove, nb_add, nb_swaps = get_number_of_modifications(data_hdfs, data_oracle)
nb_modifications = nb_remove + nb_add + nb_swaps
nb_final_classes = len(data_oracle)

print('Saving metrics to HDFS...')
save_metrics_to_hdfs(
    client, 
    docs_predicted=nb_documents_predicted, 
    docs_validated=nb_documents_validated, 
    pctg_change=pctg_change, 
    removals=nb_remove, 
    additions=nb_add, 
    swaps=nb_swaps, 
    modifications=nb_modifications, 
    total_predictions=nb_final_classes,
    classification_report=report)
