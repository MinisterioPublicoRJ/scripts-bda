from base import spark
from decouple import config

import pandas as pd
import sqlite3

URL_ORACLE_SERVER = config('URL_ORACLE_SERVER')
USER_ORACLE = config('USER_ORACLE')
PASSWD_ORACLE = config('PASSWD_ORACLE')

conn = sqlite3.connect("dev_desaparecimentos.db")


df = spark.read.format("jdbc") \
.option("url", URL_ORACLE_SERVER) \
.option("dbtable", "(SELECT * FROM SILD.SILD_DESAPARE_MOT_DECLARADO) t") \
.option("user", USER_ORACLE) \
.option("password", PASSWD_ORACLE) \
.option("driver", "oracle.jdbc.driver.OracleDriver") \
.load()
df = df.toPandas()
for col in df.columns:
    try:
        df[col] = df[col].astype(int)
    except:
        df[col] = df[col].astype(unicode)
df.astype(unicode).to_sql(name='SILD_DESAPARE_MOT_DECLARADO_TEST', con=conn, if_exists='replace', index=False)

df = spark.read.format("jdbc") \
.option("url", URL_ORACLE_SERVER) \
.option("dbtable", "(SELECT * FROM SILD.SILD_SINDICANCIA) t") \
.option("user", USER_ORACLE) \
.option("password", PASSWD_ORACLE) \
.option("driver", "oracle.jdbc.driver.OracleDriver") \
.load()
df = df.toPandas()
for col in df.columns:
    try:
        df[col] = df[col].astype(int)
    except:
        df[col] = df[col].astype(unicode)
df.astype(unicode).to_sql(name='SILD_SINDICANCIA_TEST', con=conn, if_exists='replace', index=False)

df = spark.read.format("jdbc") \
.option("url", URL_ORACLE_SERVER) \
.option("dbtable", "(SELECT * FROM SILD.SILD_SINDICANCIA_ALTERACAO) t") \
.option("user", USER_ORACLE) \
.option("password", PASSWD_ORACLE) \
.option("driver", "oracle.jdbc.driver.OracleDriver") \
.load()
df = df.toPandas()
for col in df.columns:
    try:
        df[col] = df[col].astype(int)
    except:
        df[col] = df[col].astype(unicode)
df.to_sql(name='SILD_SINDICANCIA_ALTERACAO_TEST', con=conn, if_exists='replace', index=False)