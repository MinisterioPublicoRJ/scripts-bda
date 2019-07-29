import ast
import sys
import pickle

import jaydebeapi as jdbc
from decouple import config
from hdfs import InsecureClient

from queries import (
    get_evaluate_data
)
from utils import (
    get_results_from_hdfs,
    expand_results,
    get_keys,
    get_percentage_of_change,
    get_number_of_modifications,
    generate_report,
    save_metrics_to_hdfs
)


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


print('Running Evaluate script:')
print('Connecting to HDFS and Oracle database...')
client = InsecureClient(HDFS_URL, HDFS_USER)

conn = jdbc.connect("oracle.jdbc.driver.OracleDriver",
                    URL_ORACLE_SERVER,
                    [USER_ORACLE, PASSWD_ORACLE],
                    ORACLE_DRIVER_PATH)
curs = conn.cursor()

print('Retrieving prediction results from HDFS...')
data_hdfs = get_results_from_hdfs(client, FORMATTED_HDFS_PATH)
# Results are stored as a tuple represented as a string
data_hdfs['MDEC_DK'] = data_hdfs['MDEC_DK'].apply(
    lambda x: ast.literal_eval(x))

keys = get_keys(data_hdfs, 'SNCA_DK')

print('Retrieving data from Oracle...')
# Only needs the keys that are in the predictions
data_oracle = get_evaluate_data(curs, keys)
if len(data_oracle) == 0:
    sys.exit('No validated documents available!')

print('Generating classification report...')
most_recent_date = sorted(client.list(FORMATTED_HDFS_PATH))[-1]
with client.read('{}/{}/model/mlb_binarizer.pkl'.format(
        FORMATTED_HDFS_PATH, most_recent_date)) as mlb_reader:
    mlb = pickle.loads(mlb_reader.read())

report = generate_report(data_hdfs, data_oracle, mlb)

print('Preparing HDFS data for next metrics...')
data_hdfs = expand_results(data_hdfs)

# To evaluate the predictions, only those keys that were effectively validated
# should be taken into consideration
data_hdfs = data_hdfs[data_hdfs['SNCA_DK'].isin(
    data_oracle['SNCA_DK'].unique().tolist())]

print('Calculating evaluation metrics...')
nb_documents_predicted = len(keys)
nb_documents_validated = len(get_keys(data_oracle, 'SNCA_DK'))

pctg_change = get_percentage_of_change(data_hdfs, data_oracle)

nb_remove, nb_add, nb_swaps = get_number_of_modifications(
    data_hdfs, data_oracle)
nb_modifications = nb_remove + nb_add + nb_swaps
nb_final_classes = len(data_oracle)

print('Saving metrics to HDFS...')
save_metrics_to_hdfs(
    client, FORMATTED_HDFS_PATH,
    docs_predicted=nb_documents_predicted,
    docs_validated=nb_documents_validated,
    pctg_change=pctg_change,
    removals=nb_remove,
    additions=nb_add,
    swaps=nb_swaps,
    modifications=nb_modifications,
    total_predictions=nb_final_classes,
    classification_report=report)