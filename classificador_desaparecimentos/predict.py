from base import spark

import numpy as np
import pandas as pd
import pickle
import sqlite3
import datetime

from decouple import config
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import clean_text, OneVsRestLogisticRegression, splitDataFrameList, RegexClassifier

URL_ORACLE_SERVER = config('URL_ORACLE_SERVER')
USER_ORACLE = config('USER_ORACLE')
PASSWD_ORACLE = config('PASSWD_ORACLE')

# TODO: define variables for label column name, text name, etc
NEGATIVE_CLASS_VALUE = 13
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
(SELECT A.SNCA_DK, A.SNCA_DS_FATO, 
D.MDEC_DK, D.MDEC_DS_MOTIVO_OCORRENCIA, D.MDEC_IN_SOMENTE_INSTAURACAO, E.SISI_SITUACAO_SINDICANCIA
FROM SILD.SILD_SINDICANCIA A
INNER JOIN CORP.CORP_UF B ON B.UFED_DK = A.SNCA_UFED_DK
INNER JOIN SILD.SILD_DESAPARE_MOT_DECLARADO C ON C.DMDE_SDES_DK = A.SNCA_DK
INNER JOIN SILD.SILD_MOTIVO_DECLARADO D ON D.MDEC_DK = C.DMDE_MDEC_DK
INNER JOIN SILD.SILD_SITUACAO_SINDICANCIA E ON E.SISI_DK = A.SNCA_SISI_DK
WHERE B.UFED_SIGLA = 'SP' OR B.UFED_SIGLA = 'CE') t 
"""

# TODO: read directly into pandas without using spark dataframe
df = spark.read.format("jdbc") \
.option("url", URL_ORACLE_SERVER) \
.option("dbtable", QUERY) \
.option("user", USER_ORACLE) \
.option("password", PASSWD_ORACLE) \
.option("driver", "oracle.jdbc.driver.OracleDriver") \
.load()

df = df.toPandas()

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
df[ID_COLUMN] = df[ID_COLUMN].astype(int)

df = df.groupby([ID_COLUMN, TEXT_COLUMN]).agg(lambda x: set(x)).reset_index()

X = np.array(df[TEXT_COLUMN])

mlb = pickle.load(open('mlb_binarizer.pkl', 'rb'))

reg_clf = RegexClassifier(RULES)
y_regex = reg_clf.predict(X)
y_regex = mlb.transform(y_regex)

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
X = vectorizer.transform(X)

clf = pickle.load(open('model.pkl', 'rb'))
y_logreg = clf.predict(X)

y = np.where((y_regex.sum(axis=1) > 0).reshape(-1,1), y_regex, y_logreg)

y = mlb.inverse_transform(y)

# TODO: Save model to HDFS so that it can be stored/retrieved easily
df_results = pd.DataFrame(
    np.concatenate(
        (df[ID_COLUMN].values.reshape(-1,1), np.array(y).reshape(-1,1)), 
        axis=1), 
    columns=[ID_COLUMN, LABEL_COLUMN]
)

# TODO: Talvez não valha a pena fazer uma linha para cada predição, e sim uma lista
# Isso facilita na hora de colocar os resultados no BD, não precisando verificar IDs na hora de deletar 
df_results = splitDataFrameList(df_results, LABEL_COLUMN)
df_results[LABEL_COLUMN] = df_results[LABEL_COLUMN].astype(int)

# TODO: Update rows in database accordingly
# Start updating sqlite, then we'll change this

conn = sqlite3.connect("dev_desaparecimentos.db")
cur = conn.cursor()

for label, dk in zip(df_results[LABEL_COLUMN].values, df_results[ID_COLUMN]):
    print(label, type(label))
    print(dk, type(dk))

    print('Before update desapare')
    print(cur.execute("SELECT * FROM SILD_DESAPARE_MOT_DECLARADO_TEST WHERE DMDE_SDES_DK = ?", (int(dk),)).fetchall())
    cur.execute("DELETE FROM SILD_DESAPARE_MOT_DECLARADO_TEST WHERE DMDE_SDES_DK = ?", (int(dk),))
    cur.execute("INSERT INTO SILD_DESAPARE_MOT_DECLARADO_TEST (DMDE_SDES_DK, DMDE_MDEC_DK) VALUES (?, ?)", (int(dk), int(label)))
    print('After update desapare')
    print(cur.execute("SELECT * FROM SILD_DESAPARE_MOT_DECLARADO_TEST WHERE DMDE_SDES_DK = ?", (int(dk),)).fetchall())

    print('Before update sindicancia')
    print(cur.execute("SELECT * FROM SILD_SINDICANCIA_TEST WHERE SNCA_DK = ?", (int(dk),)).fetchall())
    cur.execute("UPDATE SILD_SINDICANCIA_TEST SET SNCA_SISI_DK = 3 WHERE SNCA_DK = ?", (int(dk),))
    print('After update sindicancia')
    print(cur.execute("SELECT * FROM SILD_SINDICANCIA_TEST WHERE SNCA_DK = ?", (int(dk),)).fetchall())

    print('Before update alteracao')
    print(cur.execute("SELECT * FROM SILD_SINDICANCIA_ALTERACAO_TEST WHERE SSAL_SNCA_DK = ?", (int(dk),)).fetchall())
    cur.execute("INSERT INTO SILD_SINDICANCIA_ALTERACAO_TEST (SSAL_SNCA_DK, SSAL_DT_ALTERACAO) VALUES (?, ?)", (int(dk), datetime.datetime.now()))
    print('After update alteracao')
    print(cur.execute("SELECT * FROM SILD_SINDICANCIA_ALTERACAO_TEST WHERE SSAL_SNCA_DK = ?", (int(dk),)).fetchall())
    break