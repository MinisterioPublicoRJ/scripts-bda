import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from utils import _update_impala_table
import argparse
import params_table


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
                .builder \
                .appName("Load Data to another schema") \
                .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
                .enableHiveSupport() \
                .getOrCreate()

    tables_opt = params_table.params['tables']

    for table in tables_opt:

        schema_from = options['schema_from']
        schema_to = options['schema_to']
        
        table_df = spark.table("{}.{}".format(schema_from, table['table_name']))

        table_to = "{}.{}".format(schema_to, table['table_name'])

        if table['partition']:
            table_df.write.partitionBy(table['partition']).mode("overwrite").saveAsTable(table_to)
        else:
            table_df.write.mode("overwrite").saveAsTable(table_to)

        _update_impala_table(table_to, options['impala_host'], options['impala_port'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load Data to another schema")
    parser.add_argument('-f','--schemaFrom', metavar='schemaFrom', type=str, help='')
    parser.add_argument('-t','--schemaTo', metavar='schemaTo', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    args = parser.parse_args()

    options = {
                    'schema_from': args.schemaFrom, 
                    'schema_to': args.schemaTo,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort
                }

    execute_process(options)