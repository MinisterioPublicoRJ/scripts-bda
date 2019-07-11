# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import jaydebeapi as jdbc
import glob

from datetime import datetime
from decouple import config
from utils import clean_text, OneVsRestLogisticRegression, RegexClassifier
from hdfs import InsecureClient
from queries import (get_predict_data, 
                     set_module_and_client, 
                     get_max_dk, 
                     update_atividade_sindicancia, 
                     update_motivo_declarado)

URL_ORACLE_SERVER = config('URL_ORACLE_SERVER')
USER_ORACLE = config('USER_ORACLE')
PASSWD_ORACLE = config('PASSWD_ORACLE')
ORACLE_DRIVER_PATH = config('ORACLE_DRIVER_PATH')
ROBOT_NAME = config('ROBOT_NAME')
ROBOT_NUMBER = config('ROBOT_NUMBER')
HDFS_URL = config('HDFS_URL')
HDFS_USER = config('HDFS_USER')
HDFS_MODEL_DIR = config('HDFS_MODEL_DIR')

client = InsecureClient(HDFS_URL, user=HDFS_USER)

ID_COLUMN = 'SNCA_DK'
TEXT_COLUMN = 'SNCA_DS_FATO'
LABEL_COLUMN = 'MDEC_DK'
UFED_DK = 33

RULES = {
    2:  ['USUARI[OA] DE (DROGA|ENTORPECENTE)S?',
         'ALCOOLATRA',
         'COCAINA',
         'VICIAD[OA]',
         'DEPENDENTE QUIMICO',
         'MACONHA',
         'ALCOOL',
         'CRACK'],
    5:  ['DEPRESSAO',
         'ESQUI[ZS]OFRENIA',
         'ESQUI[ZS]OFRENIC[OA]',
         'ALZHEIMER',
         '(DOENCA|TRANSTORNO|PROBLEMA|DISTURBIO)S? MENTA(L|IS)'],
    4:  [' TRAFICO',
         'TRAFICANTES'],
    20: ['ABORDAD[OA] (POR POLICIAIS|PELA POLICIA)'],
    3:  ['FORTES CHUVAS',
         'TEMPESTADE',
         'ENXURRADA',
         'DESLIZAMENTO',
         'ROMPIMENTO D[EA] BARRAGEM',
         'SOTERRAD[OA]'],
    6:  [' RAPTOU ',
         ' RAPTAD[OA] ',
         'SEQUESTROU?',
         'SEQUESTRAD[OA]']
}

print('Running predict script:')
print('Querying database...')
conn = jdbc.connect("oracle.jdbc.driver.OracleDriver", 
                    URL_ORACLE_SERVER, 
                    [USER_ORACLE, PASSWD_ORACLE], 
                    ORACLE_DRIVER_PATH)
curs = conn.cursor()

df = get_predict_data(curs, UFED_DK=UFED_DK)

print('Preparing data...')
df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
df[ID_COLUMN] = df[ID_COLUMN].astype(int)

df = df.groupby([ID_COLUMN, TEXT_COLUMN])\
       .agg(lambda x: set(x))\
       .reset_index()

X = np.array(df[TEXT_COLUMN])

print('Loading models...')
formatted_hdfs_path = "/".join(HDFS_MODEL_DIR.split('/')[5:])
most_recent_date = sorted(client.list(formatted_hdfs_path))[-1]
with client.read('{}/{}/mlb_binarizer.pkl'.format(formatted_hdfs_path, most_recent_date)) as mlb_reader:
    mlb = pickle.loads(mlb_reader.read())
with client.read('{}/{}/vectorizer.pkl'.format(formatted_hdfs_path, most_recent_date)) as vectorizer_reader:
    vectorizer = pickle.loads(vectorizer_reader.read())
with client.read('{}/{}/model.pkl'.format(formatted_hdfs_path, most_recent_date)) as clf_reader:
    clf = pickle.loads(clf_reader.read())

print('Predicting...')
reg_clf = RegexClassifier(RULES)
y_regex = reg_clf.predict(X)
y_regex = mlb.transform(y_regex)

X = vectorizer.transform(X)
y_logreg = clf.predict(X)

y = np.where((y_regex.sum(axis=1) > 0).reshape(-1,1),
              y_regex,
              y_logreg)
y = mlb.inverse_transform(y)

df_results = pd.DataFrame(
    np.concatenate(
        (df[ID_COLUMN].values.reshape(-1,1), np.array(y).reshape(-1,1)),
        axis=1),
    columns=[ID_COLUMN, LABEL_COLUMN]
)

# Write results to HDFS
print('Writing results to HDFS...')
formatted_hdfs_path = "/".join(HDFS_MODEL_DIR.split('/')[5:])
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
client.write('{}/{}/results_{}.csv'.format(formatted_hdfs_path, most_recent_date, current_time), 
             df_results.to_csv(index=False), 
             overwrite=True)

# Should only commit everything at the end, in a single transaction
conn.jconn.setAutoCommit(False)
set_module_and_client(curs, 'DUNANT IA')

max_atsd_dk = get_max_dk(curs, 
                         table_name='SILD.SILD_ATIVIDADE_SINDICANCIA', 
                         column_name='ATSD_DK')

print('Writing results to tables...')
for labels, snca_dk in zip(df_results[LABEL_COLUMN].values, df_results[ID_COLUMN]):
    break
    #max_atsd_dk += 1
    #update_atividade_sindicancia(curs, max_atsd_dk, snca_dk, ROBOT_NAME, ROBOT_NUMBER)
    #update_motivo_declarado(curs, snca_dk, labels)

conn.commit()

curs.close()
conn.close() 
print('Done!')