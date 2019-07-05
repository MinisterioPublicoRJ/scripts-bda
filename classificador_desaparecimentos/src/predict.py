# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import jaydebeapi as jdbc

from decouple import config
from utils import clean_text, OneVsRestLogisticRegression, RegexClassifier
from hdfs import InsecureClient

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

RULES = {
    2: ['USUARI[OA] DE (DROGA|ENTORPECENTE)S?', 'ALCOOLATRA', 'COCAINA', 'VICIAD[OA]', 'DEPENDENTE QUIMICO', 'MACONHA', 'ALCOOL', 'CRACK'],
    5: ['DEPRESSAO', 'ESQUI[ZS]OFRENIA', 'ESQUI[ZS]OFRENIC[OA]', 'ALZHEIMER', 
                                '(DOENCA|TRANSTORNO|PROBLEMA|DISTURBIO)S? MENTA(L|IS)'],
    4: [' TRAFICO', 'TRAFICANTES'],
    20: ['ABORDAD[OA] (POR POLICIAIS|PELA POLICIA)'],
    3: ['FORTES CHUVAS', 'TEMPESTADE', 'ENXURRADA', 'DESLIZAMENTO', 'ROMPIMENTO D[EA] BARRAGEM', 'SOTERRAD[OA]'],
    6: [' RAPTOU ', ' RAPTAD[OA] ', 'SEQUESTROU?', 'SEQUESTRAD[OA]']
}


QUERY = """
SELECT DISTINCT B.SNCA_DK, B.SNCA_DS_FATO, D.DMDE_MDEC_DK
FROM SILD.SILD_ATIVIDADE_SINDICANCIA A
INNER JOIN SILD.SILD_SINDICANCIA B ON A.ATSD_SNCA_DK = B.SNCA_DK
INNER JOIN SILD.SILD_DESAPARE_MOT_DECLARADO D ON D.DMDE_SDES_DK = B.SNCA_DK
WHERE A.ATSD_TPSN_DK = 2 AND B.SNCA_UFED_DK = 33
AND NOT EXISTS (SELECT ATSD_SNCA_DK 
FROM SILD.SILD_ATIVIDADE_SINDICANCIA B WHERE B.ATSD_SNCA_DK = A.ATSD_SNCA_DK AND B.ATSD_TPSN_DK = 22)
"""

print('Running predict script:')
print('Querying database...')
conn = jdbc.connect("oracle.jdbc.driver.OracleDriver", URL_ORACLE_SERVER, [USER_ORACLE, PASSWD_ORACLE], ORACLE_DRIVER_PATH)
curs = conn.cursor()
curs.execute(QUERY)

columns = [desc[0] for desc in curs.description]
df = pd.DataFrame(curs.fetchall(), columns=columns)

print('Preparing data...')
df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
df[ID_COLUMN] = df[ID_COLUMN].astype(int)

df = df.groupby([ID_COLUMN, TEXT_COLUMN]).agg(lambda x: set(x)).reset_index()

X = np.array(df[TEXT_COLUMN])

print('Loading models...')
formatted_hdfs_path = "/".join(HDFS_MODEL_DIR.split('/')[5:])
with client.read('{}/mlb_binarizer.pkl'.format(formatted_hdfs_path)) as mlb_reader:
    mlb = pickle.loads(mlb_reader.read())
with client.read('{}/vectorizer.pkl'.format(formatted_hdfs_path)) as vectorizer_reader:
    vectorizer = pickle.loads(vectorizer_reader.read())
with client.read('{}/model.pkl'.format(formatted_hdfs_path)) as clf_reader:
    clf = pickle.loads(clf_reader.read())

print('Predicting...')
reg_clf = RegexClassifier(RULES)
y_regex = reg_clf.predict(X)
y_regex = mlb.transform(y_regex)

X = vectorizer.transform(X)

y_logreg = clf.predict(X)

y = np.where((y_regex.sum(axis=1) > 0).reshape(-1,1), y_regex, y_logreg)

y = mlb.inverse_transform(y)

df_results = pd.DataFrame(
    np.concatenate(
        (df[ID_COLUMN].values.reshape(-1,1), np.array(y).reshape(-1,1)),
        axis=1),
    columns=[ID_COLUMN, LABEL_COLUMN]
)

# Should only commit everything at the end, in a single transaction
conn.jconn.setAutoCommit(False)

curs.execute("CALL dbms_application_info.set_module(?, ?)", ('SILD', 'Funcionalidade'))
curs.execute("CALL dbms_application_info.set_client_info(?)", ('modelo_ia_teste',))
curs.execute("SELECT MAX(ATSD_DK) FROM SILD.SILD_ATIVIDADE_SINDICANCIA")
max_atsd_dk = int(curs.fetchall()[0][0]))

ATIV_SINDICANCIA_QUERY = """
    INSERT INTO SILD.SILD_ATIVIDADE_SINDICANCIA
    (ATSD_DK, ATSD_SNCA_DK, ATSD_TPSN_DK, ATSD_DT_REGISTRO, ATSD_DS_MOTIVO_ATIVIDADE, ATSD_NM_RESP_CTRL, ATSD_CPF_RESP_CTRL)
    VALUES (?, ?, 23, SYSDATE, 'CLASSIFICAÇÃO FEITA PELO ROBÔ', ?, ?)
"""

DELETE_MOT_DECLARADO_QUERY = """
    DELETE FROM SILD.SILD_DESAPARE_MOT_DECLARADO WHERE DMDE_SDES_DK = ?
"""

INSERT_MOT_DECLARADO_QUERY = """
    INSERT INTO SILD.SILD_DESAPARE_MOT_DECLARADO (DMDE_SDES_DK, DMDE_MDEC_DK) VALUES (?, ?)
"""

print('Writing results to tables')
for labels, snca_dk in zip(df_results[LABEL_COLUMN].values, df_results[ID_COLUMN]):

    max_atsd_dk += 1
    curs.execute(ATIV_SINDICANCIA_QUERY, (max_atsd_dk, int(snca_dk), ROBOT_NAME, ROBOT_NUMBER))
    
    curs.execute(DELETE_MOT_DECLARADO_QUERY, (int(snca_dk),))
    for label in labels: 
        curs.execute(INSERT_MOT_DECLARADO_QUERY, (int(snca_dk), int(label)))

conn.commit()
curs.close()
conn.close() 