# -*- coding: utf-8 -*-
import sys
import pickle

import numpy as np
import pandas as pd
import jaydebeapi as jdbc
from datetime import datetime
from decouple import config
from hdfs import InsecureClient

from utils import (
    clean_text,
    RegexClassifier
)
from queries import (
    get_predict_data,
    set_module_and_client,
    get_max_dk,
    update_atividade_sindicancia,
    update_motivo_declarado
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
UPDATE_TABLES = config('UPDATE_TABLES', cast=bool)
ONLY_NULL = config('ONLY_NULL', cast=bool)
START_DATE = config('START_DATE', default='')
END_DATE = config('END_DATE', default='')
UFED_DK = config('UFED_DK', default=None)
MAX_CLASSES = config('MAX_CLASSES', default=3, cast=int)
THRESHOLD = config('THRESHOLD', default=0.5, cast=float)

ID_COLUMN = 'SNCA_DK'
TEXT_COLUMN = 'SNCA_DS_FATO'
LABEL_COLUMN = 'MDEC_DK'

RULES = {
    2:  [('(?<!NAO E )(?<!NEM E )(?<!TAMPOUCO E )'
          'USUARI[OA] DE (DROGA|ENTORPECENTE)S?'),
         'ALCOOLATRA',
         'COCAINA',
         ('(?<!NAO E )(?<!NEM E )(?<!TAMPOUCO E )'
          'VICIAD[OA]'),
         ('(?<!NAO E )(?<!NEM E )(?<!TAMPOUCO E )'
          'DEPENDENTE QUIMICO'),
         'MACONHA',
         'ALCOOL',
         'CRACK'],
    5:  ['DEPRESSAO',
         'ESQUI[ZS]OFRENIA',
         'ESQUI[ZS]OFRENIC[OA]',
         'ALZHEIMER',
         'PROBLEMAS? DE ESQUECIMENTO',
         ('(?<!NAO POSSUI )(?<!NEM POSSUI )(?<!TAMPOUCO POSSUI )'
         '(?<!NAO TEM )(?<!NEM TEM )(?<!TAMPOUCO TEM )'
         '(DOENCA|TRANSTORNO|PROBLEMA|DISTURBIO)S? MENTA(L|IS)')],
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
         'SEQUESTRAD[OA]'],
    13: ['^[\w]*\s?SAIU DE CASA E NAO RETORNOU\.?$',
         '^#N/D$',
         '^(INICIALMENTE )?SEM MOTIVO APARENTE\.?$']
}


print('Running predict script:\n')

print('Connecting to HDFS and to database...')
client = InsecureClient(HDFS_URL, user=HDFS_USER)

conn = jdbc.connect("oracle.jdbc.driver.OracleDriver",
                    URL_ORACLE_SERVER,
                    [USER_ORACLE, PASSWD_ORACLE],
                    ORACLE_DRIVER_PATH)
curs = conn.cursor()

print('Querying database...')
df = get_predict_data(curs, UFED_DK=UFED_DK, only_null_class=ONLY_NULL,
                      start_date=START_DATE, end_date=END_DATE)

print('Preparing data...')
df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

df = df.groupby([ID_COLUMN, TEXT_COLUMN])\
       .agg(lambda x: set(x))\
       .reset_index()

nb_new_documents = len(df)
if nb_new_documents == 0:
    print('No new data to predict!')
    sys.exit()
else:
    print('{} new documents to predict.\n'.format(nb_new_documents))

X = np.array(df[TEXT_COLUMN])

print('Loading models...')
formatted_hdfs_path = "/".join(HDFS_MODEL_DIR.split('/')[5:])
most_recent_date = sorted(client.list(formatted_hdfs_path))[-1]
with client.read('{}/{}/model/mlb_binarizer.pkl'.format(
        formatted_hdfs_path, most_recent_date)) as mlb_reader:
    mlb = pickle.loads(mlb_reader.read())
with client.read('{}/{}/model/vectorizer.pkl'.format(
        formatted_hdfs_path, most_recent_date)) as vectorizer_reader:
    vectorizer = pickle.loads(vectorizer_reader.read())
with client.read('{}/{}/model/model.pkl'.format(
        formatted_hdfs_path, most_recent_date)) as clf_reader:
    clf = pickle.loads(clf_reader.read())

print('Predicting...')
reg_clf = RegexClassifier(RULES)
y_regex = reg_clf.predict(X)
y_regex = mlb.transform(y_regex)

X = vectorizer.transform(X)
y_logreg = clf.predict(X, threshold=THRESHOLD, max_classes=MAX_CLASSES)

y = np.where(
    (y_regex.sum(axis=1) > 0).reshape(-1, 1),
    y_regex,
    y_logreg)
y = mlb.inverse_transform(y)

# y has to be given in str format here, so as to store the results as tuples
# even if they have a single element
df_results = pd.DataFrame(
    np.concatenate(
        (df[ID_COLUMN].values.reshape(-1, 1),
            np.array([str(p) for p in y]).reshape(-1, 1)),
        axis=1),
    columns=[ID_COLUMN, LABEL_COLUMN]
)

print('Writing results to HDFS...')
formatted_hdfs_path = "/".join(HDFS_MODEL_DIR.split('/')[5:])
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
client.write(
    '{}/{}/results/{}.csv'.format(formatted_hdfs_path,
                                  most_recent_date,
                                  current_time),
    df_results.to_csv(index=False),
    overwrite=True)

# Should only commit everything at the end, in a single transaction
conn.jconn.setAutoCommit(False)

set_module_and_client(curs, 'DUNANT IA')

max_atsd_dk = get_max_dk(curs,
                         table_name='SILD.SILD_ATIVIDADE_SINDICANCIA',
                         column_name='ATSD_DK')

# Some applications of the model should not update the database tables
if UPDATE_TABLES:
    print('Writing results to tables...')
    for labels, snca_dk in zip(y, df[ID_COLUMN].values):
        max_atsd_dk += 1
        update_motivo_declarado(curs, snca_dk, labels)
        update_atividade_sindicancia(
            curs, max_atsd_dk, snca_dk, ROBOT_NAME, ROBOT_NUMBER)

    conn.commit()
else:
    print('UPDATE_TABLES is False, not updating tables...')

curs.close()
conn.close()
print('Done!')
