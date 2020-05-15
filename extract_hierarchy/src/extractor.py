#-*-coding:utf-8-*-
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import *
from utils import _update_impala_table
import argparse

import pyspark


def get_descendants(line, table):
    result = []
    for line_work in table:
        if line_work['ID_PAI'] == line['ID']:
            result.append(int(line_work['ID']))
            result.extend(get_descendants(line_work, table))
    return result


def get_hierarchy(line, table, column):
    result = line[column]
    for line_work in table:
        if line_work['ID'] == line['ID_PAI']:
            result = get_hierarchy(line_work, table, column) + ' > ' + result
    return result


def create_hierarchical_table(spark, dataframe, table_name, column):
    for line in dataframe:
        line['ID'] = int(line['ID'])
        line['ID_PAI'] = int(line['ID_PAI']) if line['ID_PAI'] else None
        #line['ID_DESCENDENTES'] = ', '.join(str(id) for id in get_descendants(line, dataframe))
        line['HIERARQUIA'] = get_hierarchy(line, dataframe, column)

    table_df = spark.createDataFrame(dataframe)
    table_df.coalesce(20).write.format('parquet').saveAsTable(table_name, mode='overwrite')


def execute_process(options):

    spark = pyspark.sql.session.SparkSession\
            .builder\
            .appName("tabelas_dominio")\
            .enableHiveSupport()\
            .getOrCreate()

    sc = spark.sparkContext

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

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

    assuntos = map(
        lambda row: row.asDict(),
        spark.table("{}.mcpr_assunto".format(schema_exadata)).select(
            col('ASSU_DK').alias('ID'),
            col('ASSU_NM_ASSUNTO').alias('NOME'),
            col('ASSU_DESCRICAO').alias('DESCRICAO'),
            col('ASSU_TX_DISPOSITIVO_LEGAL').alias('ARTIGO_LEI'),
            col('ASSU_ASSU_DK').alias('ID_PAI')
        ).collect()
    )
    
    table_name = "{}.mmps_tp_andamento".format(schema_exadata_aux)
    create_hierarchical_table(spark, andamentos, table_name, 'DESCRICAO')
    print('andamentos gravados')

    table_name = "{}.mmps_classe_docto".format(schema_exadata_aux)
    create_hierarchical_table(spark, classes, table_name, 'DESCRICAO')
    print('classes gravados')

    table_name = "{}.mmps_assunto_docto".format(schema_exadata_aux)
    create_hierarchical_table(spark, assuntos, table_name, 'NOME')
    print('assuntos gravados')

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table acervo")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort
                }

    execute_process(options)