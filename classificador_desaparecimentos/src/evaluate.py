# -*- coding: utf-8 -*-

import ast
import sys
import pickle

import jaydebeapi as jdbc
import pandas as pd
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

MOTIVOS_DICT = {
    1: 'CONFLITO INTRAFAMILIAR',
    2: 'DROGADIÇÃO',
    3: 'CATÁSTROFE',
    4: 'ENVOLVIMENTO COM TRÁFICO DE ENTORPECENTES',
    5: 'PROBLEMAS PSIQUIÁTRICOS',
    6: 'POSSÍVEL VÍTIMA DE SEQUESTRO',
    7: 'POSSÍVEL VÍTIMA DE HOMICÍDIO',
    8: 'POSSÍVEL VÍTIMA DE AFOGAMENTO',
    9: 'SUBTRAÇÃO PARA EXPLORAÇÃO ECONÔMICA',
    10: 'SUBTRAÇÃO PARA EXPLORAÇÃO SEXUAL',
    11: 'ABANDONO',
    12: 'PERDA DE CONTATO VOLUNTÁRIO',
    13: 'SEM MOTIVO APARENTE',
    14: 'AUSÊNCIA DE NOTIFICAÇÃO DE ÓBITO',
    15: 'AUSÊNCIA DE NOTIFICAÇÃO DE ENCARCERAMENTO',
    16: 'AUSÊNCIA DE NOTIFICAÇÃO DE INSTITUCIONALIZAÇÃO',
    17: 'VÍTIMA DE SEQUESTRO',
    18: 'OCULTAÇÃO DE CADÁVER',
    19: 'VÍTIMA DE AFOGAMENTO',
    20: 'PRISÃO/APREENSÃO',
    21: 'POSSÍVEL VÍTIMA DE FEMINICÍDIO'
}


print('Running Evaluate script:')
print('Connecting to HDFS and Oracle database...')
client = InsecureClient(HDFS_URL, HDFS_USER)

conn = jdbc.connect("oracle.jdbc.driver.OracleDriver",
                    URL_ORACLE_SERVER,
                    [USER_ORACLE, PASSWD_ORACLE],
                    ORACLE_DRIVER_PATH)
curs = conn.cursor()

model_dates = sorted(client.list(FORMATTED_HDFS_PATH))
validated_datasets = []
classified_datasets = []

for model_date in model_dates:
    try:
        data_hdfs = get_results_from_hdfs(client, FORMATTED_HDFS_PATH, model_date=model_date)
    except:
        continue
    # Results are stored as a tuple represented as a string
    data_hdfs['MDEC_DK'] = data_hdfs['MDEC_DK'].apply(
        lambda x: ast.literal_eval(x))

    keys = get_keys(data_hdfs, 'SNCA_DK')

    # Only needs the keys that are in the predictions
    data_oracle = get_evaluate_data(curs, keys)

    data_hdfs = expand_results(data_hdfs)
    data_hdfs['DT_MODELO'] = model_date
    data_oracle['DT_MODELO'] = model_date
    data_oracle['IS_VALIDATION'] = True
    data_hdfs['IS_VALIDATION'] = False

    validated_datasets.append(data_oracle)
    classified_datasets.append(data_hdfs)


df1 = pd.concat(validated_datasets, ignore_index=True)
df2 = pd.concat(classified_datasets, ignore_index=True)

result_df = pd.concat([df1, df2], ignore_index=True)
result_df['MDEC_MOTIVO'] = result_df['MDEC_DK'].apply(lambda x: MOTIVOS_DICT[x])

result_df.to_csv('results_dunant.csv', index=False)

#print(result_df.groupby(['SNCA_DK', 'IS_VALIDATION']).count())


