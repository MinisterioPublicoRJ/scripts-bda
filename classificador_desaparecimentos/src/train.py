import numpy as np
import pandas as pd
import pickle
import jaydebeapi as jdbc

from decouple import config
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import clean_text, OneVsRestLogisticRegression
from hdfs import InsecureClient

URL_ORACLE_SERVER = config('URL_ORACLE_SERVER')
USER_ORACLE = config('USER_ORACLE')
PASSWD_ORACLE = config('PASSWD_ORACLE')
ORACLE_DRIVER_PATH = config('ORACLE_DRIVER_PATH')
HDFS_URL = config('HDFS_URL')
HDFS_USER = config('HDFS_USER')
HDFS_MODEL_DIR = config('HDFS_MODEL_DIR')

client = InsecureClient(HDFS_URL, user=HDFS_USER)

NEGATIVE_CLASS_VALUE = 13
ID_COLUMN = 'SNCA_DK'
TEXT_COLUMN = 'SNCA_DS_FATO'
LABEL_COLUMN = 'DMDE_MDEC_DK'

QUERY = """
SELECT DISTINCT B.SNCA_DK, B.SNCA_DS_FATO, D.DMDE_MDEC_DK
FROM SILD.SILD_ATIVIDADE_SINDICANCIA A
INNER JOIN SILD.SILD_SINDICANCIA B ON A.ATSD_SNCA_DK = B.SNCA_DK
INNER JOIN SILD.SILD_DESAPARE_MOT_DECLARADO D ON D.DMDE_SDES_DK = B.SNCA_DK
WHERE A.ATSD_TPSN_DK = 22 AND B.SNCA_UFED_DK = 33
"""

print('Running train script:')
print('Querying database...')
conn = jdbc.connect("oracle.jdbc.driver.OracleDriver", URL_ORACLE_SERVER, [USER_ORACLE, PASSWD_ORACLE], ORACLE_DRIVER_PATH)
curs = conn.cursor()
curs.execute(QUERY)

columns = [desc[0] for desc in curs.description]
df = pd.DataFrame(curs.fetchall(), columns=columns)

print('Preparing data...')
df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

df = df.groupby(TEXT_COLUMN).agg(lambda x: set(x)).reset_index()

mlb = MultiLabelBinarizer()
y = df[LABEL_COLUMN]
y = mlb.fit_transform(y)

NEGATIVE_COLUMN_INDEX = np.where(mlb.classes_ == NEGATIVE_CLASS_VALUE)[0][0]
y[:,NEGATIVE_COLUMN_INDEX] = y[:,NEGATIVE_COLUMN_INDEX]*~((y.sum(axis=1) > 1) & (y[:,NEGATIVE_COLUMN_INDEX] == 1))

X = np.array(df[TEXT_COLUMN])

vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.6, min_df=5)
X = vectorizer.fit_transform(X)

print('Fitting model to data...')
clf = OneVsRestLogisticRegression(negative_column_index=NEGATIVE_COLUMN_INDEX, class_weight='balanced')
clf.fit(X, y)

print('Saving to HDFS...')
mlb_pickle = pickle.dumps(mlb)
vectorizer_pickle = pickle.dumps(vectorizer)
clf_pickle = pickle.dumps(clf)

formatted_hdfs_path = "/".join(HDFS_MODEL_DIR.split('/')[5:])

client.write('{}/mlb_binarizer.pkl'.format(formatted_hdfs_path), mlb_pickle, overwrite=True)
client.write('{}/vectorizer.pkl'.format(formatted_hdfs_path), vectorizer_pickle, overwrite=True)
client.write('{}/model.pkl'.format(formatted_hdfs_path), clf_pickle, overwrite=True)

print('Done!')
