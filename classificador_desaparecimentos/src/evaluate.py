# -*- coding: utf-8 -*-

import ast
import httplib2

import jaydebeapi as jdbc
import pandas as pd
import gspread
from decouple import config
from hdfs import InsecureClient
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe

from queries import (
    get_evaluate_data,
    get_id_sinalid
)
from utils import (
    get_results_from_hdfs,
    expand_results,
    get_keys
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
EVALUATE_SAVE_GSPREAD = config(
    'EVALUATE_SAVE_GSPREAD',
    cast=bool,
    default=False)
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
        data_hdfs = get_results_from_hdfs(
            client,
            FORMATTED_HDFS_PATH,
            model_date=model_date)
    except BaseException:
        continue
    # Results are stored as a tuple represented as a string
    data_hdfs['MDEC_DK'] = data_hdfs['MDEC_DK'].apply(
        lambda x: ast.literal_eval(x))

    keys = get_keys(data_hdfs, 'SNCA_DK')

    # Only needs the keys that are in the predictions
    data_oracle = get_evaluate_data(curs, keys)

    data_hdfs = expand_results(data_hdfs)

    # TODO: Optimization possible here, gets all keys in query each time before filtering by keys
    sinalid_id_df = get_id_sinalid(curs, keys)
    data_hdfs = pd.merge(data_hdfs, sinalid_id_df, how='left', on='SNCA_DK')

    date = '{}/{}/{}'.format(model_date[6:8], model_date[4:6], model_date[:4])
    data_hdfs['DT_MODELO'] = date
    data_oracle['DT_MODELO'] = date
    data_oracle['IS_VALIDATION'] = True
    data_hdfs['IS_VALIDATION'] = False

    validated_datasets.append(data_oracle)
    classified_datasets.append(data_hdfs)

df1 = pd.concat(validated_datasets, ignore_index=True)
df2 = pd.concat(classified_datasets, ignore_index=True)

result_df = pd.concat([df1, df2], ignore_index=True)
result_df['MDEC_MOTIVO'] = result_df['MDEC_DK'].apply(
    lambda x: MOTIVOS_DICT[x])

percents = []
for x in list(result_df[['MDEC_DK', 'MDEC_MOTIVO']].drop_duplicates().values):
    c = x[0]
    ds = x[1]
    class_preds = df2[df2['MDEC_DK'] == c]
    class_trues = df1[['SNCA_DK', 'MDEC_DK']].groupby('SNCA_DK').agg(lambda x: c in set(x)).reset_index()
    
    df = class_preds[['SNCA_DK', 'MDEC_DK']].merge(
        class_trues, 
        how='inner', 
        left_on='SNCA_DK', 
        right_on='SNCA_DK')
    if df.shape[0] != 0:
        percents.append([str(c) + ',' + ds, df[df['MDEC_DK_y'] == True].shape[0]/df.shape[0]])
    else:
        percents.append([str(c) + ',' + ds, 'Sem classificações ou validações nesta classe'])

percents = pd.DataFrame(percents)
percents.rename({1: 'PRECISAO'}, axis=1, inplace=True)
percents['MDEC_DK'], percents['MDEC_MOTIVO'] = percents[0].str.split(',').str
percents.drop(0, axis=1, inplace=True)

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'dunant_credentials-5ab80fd0863c.json',
    scopes=scope)

gc = gspread.client.Client(auth=credentials)
http = httplib2.Http(disable_ssl_certificate_validation=True, ca_certs='')
gc.auth.refresh(http)
gc.login()

sh = gc.open("results_dunant")
worksheet = sh.get_worksheet(0)
worksheet_percents = sh.get_worksheet(1)

if EVALUATE_SAVE_GSPREAD:
    set_with_dataframe(worksheet, result_df, resize=True)
    set_with_dataframe(worksheet_percents, percents, resize=True)
else:
    result_df.to_csv('results_dunant.csv', index=False)
    percents.to_csv('results_dunant_percentages.csv', index=False)
