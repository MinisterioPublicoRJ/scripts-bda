import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
import argparse


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
                .builder \
                .appName("Execute SQL") \
                .config("hive.exec.dynamic.partition.mode", "nonstrict") \
                .enableHiveSupport() \
                .getOrCreate()

    with open(options['file_name'], 'r') as file:
        sql_query = file.read().replace('\n', '')

    for sql in sql_query.split("---EOS---"): 
        table = spark.sql(sql.format(schema_exadata_aux=options['schema_exadata_aux'], schema_exadata=options['schema_exadata']))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Execute SQL")
    parser.add_argument('-f','--fileName', metavar='fileName', type=str, help='')
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    args = parser.parse_args()

    options = {
                    'file_name': args.fileName,
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort
                }

    execute_process(options)
