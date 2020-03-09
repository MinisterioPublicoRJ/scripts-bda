#-*-coding:utf-8-*-
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import *
from decouple import config

import pyspark

schema_exadata = config('SCHEMA_EXADATA')
schema_exadata_aux = config('SCHEMA_EXADATA_AUX')

def get_descendants(line, table):
    result = []
    for line_work in table:
        if line_work['ID_PAI'] == line['ID']:
            result.append(int(line_work['ID']))
            result.extend(get_descendants(line_work, table))
    return result


def get_hierarchy(line, table):
    result = line['DESCRICAO']
    for line_work in table:
        if line_work['ID'] == line['ID_PAI']:
            result = get_hierarchy(line_work, table) + ' > ' + result
    return result


def create_hierarchical_table(dataframe, table_name):
    for line in dataframe:
        line['ID'] = int(line['ID'])
        line['ID_PAI'] = int(line['ID_PAI']) if line['ID_PAI'] else None
        line['ID_DESCENDENTES'] = ', '.join(str(id) for id in get_descendants(line, dataframe))
        line['HIERARQUIA'] = get_hierarchy(line, dataframe)

    table_df = spark.createDataFrame(dataframe)
    schema_name = "{}.{}".format(schema_exadata_aux, table_name)
    table_df.coalesce(20).write.format('parquet').saveAsTable(schema_name, mode='overwrite')


spark = pyspark.sql.session.SparkSession\
        .builder\
        .appName("tabelas_dominio")\
        .enableHiveSupport()\
        .getOrCreate()

sc = spark.sparkContext

andamentos = map(
    lambda row: row.asDict(),
    spark.table("{}.mcpr_tp_andamento".format(schema_exadata)).select(
        col('TPPR_DK').alias('ID'),
        col('TPPR_DESCRICAO').alias('DESCRICAO'),
        col('TPPR_CD_TP_ANDAMENTO').alias('COD_MGP'),
        col('TPPR_TPPR_DK').alias('ID_PAI')
    ).collect()
)

classes = map(
    lambda row: row.asDict(),
    spark.table("{}.mcpr_classe_docto_mp".format(schema_exadata)).select(
        col('CLDC_DK').alias('ID'),
        col('CLDC_DS_CLASSE').alias('DESCRICAO'),
        col('CLDC_CD_CLASSE').alias('COD_MGP'),
        col('CLDC_CLDC_DK_SUPERIOR').alias('ID_PAI')
    ).collect()
)

create_hierarchical_table(andamentos, 'mmps_tp_andamento')
print('andamentos gravados')

create_hierarchical_table(classes, 'mmps_classe_docto')
print('classes gravados')