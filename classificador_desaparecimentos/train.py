from base import spark

import pandas as pd
import numpy as np
import re

from copy import deepcopy
from unidecode import unidecode
from decouple import config

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from utils import clean_text

URL_ORACLE_SERVER = config('URL_ORACLE_SERVER')
USER_ORACLE = config('USER_ORACLE')
PASSWD_ORACLE = config('PASSWD_ORACLE')

# TODO: define variables for label column name, text name, etc

QUERY = """
(SELECT A.SNCA_DK, B.UFED_SIGLA, A.SNCA_IDENTIFICADOR_SINALID, A.SNCA_DS_FATO, 
A.SNCA_DT_FATO, A.SNCA_PC_RELEVANCIA_POR_DOCTO, A.SNCA_IN_TRAFICO_PESSOAS,
D.MDEC_DK, D.MDEC_DS_MOTIVO_OCORRENCIA, D.MDEC_IN_SOMENTE_INSTAURACAO, E.SISI_SITUACAO_SINDICANCIA
FROM SILD.SILD_SINDICANCIA A
INNER JOIN CORP.CORP_UF B ON B.UFED_DK = A.SNCA_UFED_DK
INNER JOIN SILD.SILD_DESAPARE_MOT_DECLARADO C ON C.DMDE_SDES_DK = A.SNCA_DK
INNER JOIN SILD.SILD_MOTIVO_DECLARADO D ON D.MDEC_DK = C.DMDE_MDEC_DK
INNER JOIN SILD.SILD_SITUACAO_SINDICANCIA E ON E.SISI_DK = A.SNCA_SISI_DK) t 
"""

# TODO: read directly into pandas without using spark dataframe
data = spark.read.format("jdbc") \
.option("url", URL_ORACLE_SERVER) \
.option("dbtable", QUERY) \
.option("user", USER_ORACLE) \
.option("password", PASSWD_ORACLE) \
.option("driver", "oracle.jdbc.driver.OracleDriver") \
.load()

data = data.toPandas()

data['SNCA_DS_FATO'] = data['SNCA_DS_FATO'].apply(clean_text)

# TODO: Train and test should take into considered SITUACAO_SINDICANCIA
data_train = data[(data['UFED_SIGLA'] == 'SP') | (data['UFED_SIGLA'] == 'CE')]
# data_test = data[(data['UFED_SIGLA'] != 'SP') & (data['UFED_SIGLA'] != 'CE')]

df = data_train.copy()
df = df.groupby(['SNCA_DK', 'SNCA_DS_FATO', 'SNCA_DT_FATO', 'SISI_SITUACAO_SINDICANCIA', 
                  'UFED_SIGLA', 'SNCA_IDENTIFICADOR_SINALID', 'SNCA_PC_RELEVANCIA_POR_DOCTO', 
                  'SNCA_IN_TRAFICO_PESSOAS']
                ).agg({'MDEC_DK': list}).reset_index()

mlb = MultiLabelBinarizer()
y = df['MDEC_DK']
y = mlb.fit_transform(y)
y[:,12] = y[:,12]*~((y.sum(axis=1) > 1) & (y[:,12] == 1))

# df.drop('MDEC_DK', axis=1, inplace=True)

X = np.array(df['SNCA_DS_FATO'])


data = data.na.drop(subset=["SNCA_DS_FATO"])

vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.6, min_df=5)
X = vectorizer.fit_transform(X)

y = np.delete(y, 12, axis=1)

# TODO: Do a wrapper for the model so that the prediction includes "SEM MOTIVO APARENTE"
classif = OneVsRestClassifier(LogisticRegression(class_weight='balanced'))
classif.fit(X, y)

# TODO: Save model to HDFS so that it can be stored/retrieved easily

