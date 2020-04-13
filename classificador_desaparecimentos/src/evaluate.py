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
from unidecode import unidecode

from queries import (
    get_evaluate_data,
    get_id_sinalid,
    get_motivos_declarados,
)
from utils import (
    get_results_from_hdfs,
    expand_results,
    get_keys,
    parse_arguments,
)


options = parse_arguments()

URL_ORACLE_SERVER =  options['oracle_server']
USER_ORACLE =  options['oracle_user']
PASSWD_ORACLE =  options['oracle_password']
ORACLE_DRIVER_PATH =  options['oracle_driver_path']
HDFS_URL =  options['hdfs_url']
HDFS_USER =  options['hdfs_user']
HDFS_MODEL_DIR =  options['hdfs_model_dir']
EVALUATE_SAVE_GSPREAD = options['evaluate_save_gspread']
FORMATTED_HDFS_PATH = "/".join(HDFS_MODEL_DIR.split('/')[5:])


print('Running Evaluate script:')
print('Connecting to HDFS and Oracle database...')
client = InsecureClient(HDFS_URL, HDFS_USER)

conn = jdbc.connect("oracle.jdbc.driver.OracleDriver",
                    URL_ORACLE_SERVER,
                    [USER_ORACLE, PASSWD_ORACLE],
                    ORACLE_DRIVER_PATH)
curs = conn.cursor()

MOTIVOS_DICT = get_motivos_declarados(curs)

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

    # TODO: Optimization possible here,
    # gets all keys in query each time before filtering by keys
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
    lambda x: unidecode(MOTIVOS_DICT[x]))

percents = []
for x in list(result_df[['MDEC_DK', 'MDEC_MOTIVO']].drop_duplicates().values):
    c = x[0]
    ds = x[1]
    class_preds = df2[df2['MDEC_DK'] == c]
    class_trues = df1[['SNCA_DK', 'MDEC_DK']].groupby('SNCA_DK').agg(
        lambda x: c in set(x)).reset_index()

    df = class_preds[['SNCA_DK', 'MDEC_DK']].merge(
        class_trues,
        how='inner',
        left_on='SNCA_DK',
        right_on='SNCA_DK')
    if df.shape[0] != 0:
        percents.append(
            [str(c) + ',' + ds,
             df[df['MDEC_DK_y']].shape[0]/float(df.shape[0]),
             df.shape[0]])
    else:
        percents.append(
            [str(c) + ',' + ds,
             'Sem classificacoes ou validacoes nesta classe',
             'Nao se aplica'])

percents = pd.DataFrame(percents)
percents.rename({1: 'PRECISAO', 2: 'SUPORTE'}, axis=1, inplace=True)
percents['MDEC_DK'], percents['MDEC_MOTIVO'] = percents[0].str.split(',').str
percents.drop(0, axis=1, inplace=True)
percents['PRECISAO'] = percents['PRECISAO'].apply(
    lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x)

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'dunant_credentials-5ab80fd0863c.json',
    scopes=scope)

gc = gspread.client.Client(auth=credentials)
http = httplib2.Http(disable_ssl_certificate_validation=True, ca_certs='')
gc.auth.refresh(http)
gc.login()

sh1 = gc.open("results_dunant")
worksheet = sh1.get_worksheet(0)

sh2 = gc.open("results_dunant_percentages")
worksheet_percents = sh2.get_worksheet(0)

if EVALUATE_SAVE_GSPREAD:
    set_with_dataframe(worksheet, result_df, resize=True)
    set_with_dataframe(worksheet_percents, percents, resize=True)
else:
    result_df.to_csv('results_dunant.csv', index=False)
    percents.to_csv('results_dunant_percentages.csv', index=False)
