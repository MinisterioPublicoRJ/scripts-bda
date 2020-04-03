import sys
import pickle

import jaydebeapi as jdbc
import numpy as np
from datetime import datetime
from decouple import config
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from hdfs import InsecureClient

from utils import (
    clean_text,
    get_keys,
    parse_arguments,
)
from models import (
    RegexClassifier,
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

models_to_evaluate = [
    ('OneVsRestClassifier', OneVsRestClassifier(LogisticRegression())),
    ('RandomForest', OneVsRestClassifier(RandomForestClassifier()))
]

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

print('Preparing data...')
df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

df = df.sort_values(ID_COLUMN)

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

reg_clf = RegexClassifier(RULES)
y_regex = reg_clf.predict(X)
y_regex = mlb.transform(y_regex)

vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE,
                             max_df=MAX_DF,
                             min_df=MIN_DF)
X = vectorizer.fit_transform(X)

l = int(0.9*X.shape[0])
X_train, X_test = X[:l], X[l:]
y_train, y_test = y[:l], y[l:]
y_regex_train, y_regex_test = y_regex[:l], y_regex[l:]

print('Length of train: {}'.format(X_train.shape[0]))
print('Length of test: {}'.format(X_test.shape[0]))

reports = []
## Fit Model
for model_name, model in models_to_evaluate:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # If regex, then take it
    y_pred_with_regex = np.where(
        (y_regex_test.sum(axis=1) > 0).reshape(-1, 1),
        y_regex_test,
        y_pred)
    # Also try sum with regex

    reports.append(
        (   model_name + " with Regex",
            classification_report(y_test, y_pred_with_regex)
        )
    )
    reports.append(
        (   model_name,
            classification_report(y_test, y_pred)
        )
    )

with open('evaluation_reports.txt', 'w') as output:
    for model_name, report in reports:
        output.write(model_name)
        output.write(report)
        output.write('\n')



