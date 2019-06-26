from base import spark

import numpy as np
import pickle

from decouple import config
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import clean_text, OneVsRestLogisticRegression

URL_ORACLE_SERVER = config('URL_ORACLE_SERVER')
USER_ORACLE = config('USER_ORACLE')
PASSWD_ORACLE = config('PASSWD_ORACLE')

# TODO: define variables for label column name, text name, etc
NEGATIVE_CLASS_VALUE = 13
ID_COLUMN = 'SNCA_DK'
TEXT_COLUMN = 'SNCA_DS_FATO'
LABEL_COLUMN = 'MDEC_DK'

# QUERY = """
# (SELECT A.SNCA_DK, B.UFED_SIGLA, A.SNCA_IDENTIFICADOR_SINALID, A.SNCA_DS_FATO, 
# A.SNCA_DT_FATO, A.SNCA_PC_RELEVANCIA_POR_DOCTO, A.SNCA_IN_TRAFICO_PESSOAS,
# D.MDEC_DK, D.MDEC_DS_MOTIVO_OCORRENCIA, D.MDEC_IN_SOMENTE_INSTAURACAO, E.SISI_SITUACAO_SINDICANCIA
# FROM SILD.SILD_SINDICANCIA A
# INNER JOIN CORP.CORP_UF B ON B.UFED_DK = A.SNCA_UFED_DK
# INNER JOIN SILD.SILD_DESAPARE_MOT_DECLARADO C ON C.DMDE_SDES_DK = A.SNCA_DK
# INNER JOIN SILD.SILD_MOTIVO_DECLARADO D ON D.MDEC_DK = C.DMDE_MDEC_DK
# INNER JOIN SILD.SILD_SITUACAO_SINDICANCIA E ON E.SISI_DK = A.SNCA_SISI_DK) t 
# """

# TODO: Train and test should take into consideration SITUACAO_SINDICANCIA
QUERY = """
    (SELECT A.SNCA_DK, A.SNCA_DS_FATO, 
    D.MDEC_DK, D.MDEC_DS_MOTIVO_OCORRENCIA, E.SISI_SITUACAO_SINDICANCIA
    FROM SILD.SILD_SINDICANCIA A
    INNER JOIN CORP.CORP_UF B ON B.UFED_DK = A.SNCA_UFED_DK
    INNER JOIN SILD.SILD_DESAPARE_MOT_DECLARADO C ON C.DMDE_SDES_DK = A.SNCA_DK
    INNER JOIN SILD.SILD_MOTIVO_DECLARADO D ON D.MDEC_DK = C.DMDE_MDEC_DK
    INNER JOIN SILD.SILD_SITUACAO_SINDICANCIA E ON E.SISI_DK = A.SNCA_SISI_DK
    WHERE B.UFED_SIGLA != 'SP' AND B.UFED_SIGLA != 'CE') t
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

df = df.groupby(TEXT_COLUMN).agg(lambda x: set(x)).reset_index()

mlb = MultiLabelBinarizer()
y = df[LABEL_COLUMN]
y = mlb.fit_transform(y)

NEGATIVE_COLUMN_INDEX = np.where(mlb.classes_ == NEGATIVE_CLASS_VALUE)[0][0]
y[:,NEGATIVE_COLUMN_INDEX] = y[:,NEGATIVE_COLUMN_INDEX]*~((y.sum(axis=1) > 1) & (y[:,NEGATIVE_COLUMN_INDEX] == 1))

X = np.array(df[TEXT_COLUMN])

vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.6, min_df=5)
X = vectorizer.fit_transform(X)

clf = OneVsRestLogisticRegression(negative_column_index=NEGATIVE_COLUMN_INDEX, class_weight='balanced')
clf.fit(X, y)

# TODO: Save model to HDFS so that it can be stored/retrieved easily
with open('mlb_binarizer.pkl', 'wb') as binarizer_file:
    pickle.dump(mlb, binarizer_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
