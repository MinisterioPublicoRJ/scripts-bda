import sys
import pickle

import jaydebeapi as jdbc
import numpy as np
from datetime import datetime
from decouple import config
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from hdfs import InsecureClient

from utils import (
    clean_text,
    get_keys,
    parse_arguments,
)
from models import (
    OneVsRestLogisticRegression
)
from queries import (
    get_train_data,
    get_list_of_classes
)

options = parse_arguments()

URL_ORACLE_SERVER =  options['oracle_server']
USER_ORACLE =  options['oracle_user']
PASSWD_ORACLE =  options['oracle_password']
ORACLE_DRIVER_PATH =  options['oracle_driver_path']
HDFS_URL =  options['hdfs_url']
HDFS_USER =  options['hdfs_user']
HDFS_MODEL_DIR =  options['hdfs_model_dir']
START_DATE =  options['start_date']
END_DATE =  options['end_date']

NEGATIVE_CLASS_VALUE = 13
ID_COLUMN = 'SNCA_DK'
TEXT_COLUMN = 'SNCA_DS_FATO'
LABEL_COLUMN = 'DMDE_MDEC_DK'

# Vectorizer parameters
NGRAM_RANGE = (1, 3)
MAX_DF = 0.6
MIN_DF = 1


print('Running train script:')
print('Querying database...')
client = InsecureClient(HDFS_URL, user=HDFS_USER)

conn = jdbc.connect("oracle.jdbc.driver.OracleDriver",
                    URL_ORACLE_SERVER,
                    [USER_ORACLE, PASSWD_ORACLE],
                    ORACLE_DRIVER_PATH)
curs = conn.cursor()

df = get_train_data(curs,
                    start_date=START_DATE, end_date=END_DATE)

nb_documents = len(df)
if nb_documents == 0:
    print('No data to train model!')
    sys.exit()
else:
    print('{} documents available to train model.\n'.format(nb_documents))

train_keys = get_keys(df, ID_COLUMN)

print('Preparing data...')
df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

# Labels need to be grouped to be passed to the MultiLabelBinarizer
df = df.groupby(TEXT_COLUMN)\
       .agg(lambda x: set(x))\
       .reset_index()

classes = get_list_of_classes(curs)
mlb = MultiLabelBinarizer(classes)
y = df[LABEL_COLUMN]
y = mlb.fit_transform(y)

NEGATIVE_COLUMN_INDEX = np.where(mlb.classes_ == NEGATIVE_CLASS_VALUE)[0][0]
# If row has more than one class, and one of them is the null class,
# remove null class
y[:, NEGATIVE_COLUMN_INDEX] = y[:, NEGATIVE_COLUMN_INDEX]*~(
    (y.sum(axis=1) > 1) & (y[:, NEGATIVE_COLUMN_INDEX] == 1))

X = np.array(df[TEXT_COLUMN])

vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE,
                             max_df=MAX_DF,
                             min_df=MIN_DF)
X = vectorizer.fit_transform(X)

print('Fitting model to data...')
clf = OneVsRestLogisticRegression(negative_column_index=NEGATIVE_COLUMN_INDEX,
                                  class_weight='balanced')
clf.fit(X, y)

print('Saving to HDFS...')
mlb_pickle = pickle.dumps(mlb)
vectorizer_pickle = pickle.dumps(vectorizer)
clf_pickle = pickle.dumps(clf)

formatted_hdfs_path = "/".join(HDFS_MODEL_DIR.split('/')[5:])
current_time = datetime.now().strftime('%Y%m%d%H%M%S')

client.write(
    '{}/{}/model/mlb_binarizer.pkl'.format(formatted_hdfs_path, current_time),
    mlb_pickle,
    overwrite=True
)
client.write(
    '{}/{}/model/vectorizer.pkl'.format(formatted_hdfs_path, current_time),
    vectorizer_pickle,
    overwrite=True
)
client.write(
    '{}/{}/model/model.pkl'.format(formatted_hdfs_path, current_time),
    clf_pickle,
    overwrite=True
)

keys_string = 'SNCA_DK\n' + "\n".join([str(int(k)) for k in train_keys])
client.write(
    '{}/{}/model/train_keys.csv'.format(formatted_hdfs_path, current_time),
    keys_string,
    overwrite=True
)

print('Done!')
