import os

import pyspark
import argparse

from pyspark.sql.functions import year
from generic_utils import execute_compute_stats

def execute_process(args):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_regcivil_detran") \
            .config("hive.exec.dynamic.partition.mode", "nonstrict") \
            .enableHiveSupport() \
            .getOrCreate()
    
    schema_staging = args.schemaStaging
    schema_detran = args.schemaDetran

    table_detran = spark.table("{}.detran_regcivil".format(schema_staging))

    table_name = "{}.tb_regcivil".format(schema_detran)

    table_detran = table_detran.withColumn("year", year('dt_expedicao_carteira'))

    table_detran.write.partitionBy('year').format("hive").mode("overwrite").saveAsTable(table_name)

    execute_compute_stats(table_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="criar tabela regcivil detran")
    parser.add_argument('-st','--schemaStaging', metavar='schemaStaging', type=str, help='')
    parser.add_argument('-sd','--schemaDetran', metavar='schemaDetran', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    args = parser.parse_args()

    execute_process(args)