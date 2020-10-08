#-*-coding:utf-8-*-
import argparse
import pyspark

from pyspark.sql.types import IntegerType
from pyspark.sql.functions import *

from generic_utils import execute_compute_stats


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


def create_hierarchical_table(spark, dataframe, table_name):
    for line in dataframe:
        line['ID'] = int(line['ID'])
        line['ID_PAI'] = int(line['ID_PAI']) if line['ID_PAI'] else None
        line['ID_DESCENDENTES'] = ', '.join(str(id) for id in get_descendants(line, dataframe))
        line['HIERARQUIA'] = get_hierarchy(line, dataframe)

    table_df = spark.createDataFrame(dataframe)
    table_df.coalesce(20).write.format('parquet').saveAsTable(table_name, mode='overwrite')

    execute_compute_stats(table_name)

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

    table_name = "{}.mmps_tp_andamento".format(schema_exadata_aux)
    create_hierarchical_table(spark, andamentos, table_name)
    print('andamentos gravados')

    table_name = "{}.mmps_classe_docto".format(schema_exadata_aux)
    create_hierarchical_table(spark, classes, table_name)
    print('classes gravados')


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