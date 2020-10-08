import os

import argparse
import params_table
import pyspark
#import unicodedata
import unidecode
from hdfs.ext.kerberos import KerberosClient
from pyspark.sql.functions import year, col, regexp_replace
from generic_utils import execute_compute_stats
from solr_log import send_log, ERROR, SUCCESS, SUCCESS_MESSAGE, ERROR_MESSAGE


def trait_columns_name(value):
    value = value.replace(" ", "_").replace("\r", "")
    u = unicode(value, "utf-8")
    return unidecode.unidecode(u)

def remove_break_lines(df, (col_name, dtype)):
    
    df = df.withColumn(col_name, regexp_replace(col(col_name),'\r','').cast(dtype))

    return df

def check_type(df, (col_name, dtype)):
    
    if 'decimal' == dtype:
        df = df.withColumn(col_name, regexp_replace(col(col_name),'^,','')) \
            .withColumn(col_name, regexp_replace(col(col_name),',','.').cast('decimal(10,2)'))
    else:
        df = df.withColumn(col_name, col(col_name).cast(dtype))
    return df

def export_to_postgres(df, args, tablePostgres):

    properties = {
        "user": args.jdbcUser,
        "password": args.jdbcPassword, 
        "driver": "org.postgresql.Driver",
        }

    df.write.jdbc(
        "jdbc:postgresql://{jdbcServer}:5432/{database}".format(
            jdbcServer=args.jdbcServer, 
            database=args.jdbcDatabase),
        tablePostgres,
        properties=properties, 
        mode="overwrite")

def execute_process(args):

    app_name = "criar_tabela_tce"
    spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName(app_name) \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .enableHiveSupport() \
        .getOrCreate()

    client = KerberosClient(args.webHdfs)

    hdfs_files = client.list(args.pathDirectoryBase)

    for directory in hdfs_files:

        try:

            actual_directory = args.pathDirectoryBase + directory

            df = spark.read.text(actual_directory)

            if not df.rdd.isEmpty():

                # df = spark.read.load(actual_directory, format="csv", multiLine=True,
			    #                     sep=args.delimiter, inferSchema=True, header=True)
                
                columns_types = params_table.table_columns_type[directory]

                df = spark.read.option("quote", "\"") \
                    .option("escape", "\"") \
                    .load(actual_directory, format="csv", sep=args.delimiter, header=True)

                columns = [trait_columns_name(column_name) for column_name in df.columns]
                
                df = df.toDF(*columns)
                
                df = reduce(check_type, columns_types, df)

                #df = reduce(remove_break_lines, df.dtypes, df)
                    
                table_hive = "{}.{}".format(args.schemaHive, directory)
                
                table_postgres = "{}.{}".format(args.schemaPostgres, directory)

                df.write.mode("overwrite").format("parquet").saveAsTable(table_hive)

                spark.sql("ANALYZE TABLE {} COMPUTE STATISTICS".format(table_hive))
                
                execute_compute_stats(table_hive)

                export_to_postgres(df, args, table_postgres)

                send_log(SUCCESS_MESSAGE.format(directory), app_name, SUCCESS, args.solrServer)

        except Exception as message:
            send_log(ERROR_MESSAGE.format(directory, message), app_name, ERROR, args.solrServer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="criar tabela tce")
    parser.add_argument('-sh', '--schemaHive',
                        metavar='schemaHive', type=str, help='')
    parser.add_argument('-wh', '--webHdfs',
                        metavar='webHdfs', type=str, help='')
    parser.add_argument('-u', '--userWebHdfs',
                        metavar='userWebHdfs', type=str, help='')
    parser.add_argument('-p', '--pathDirectoryBase',
                        metavar='pathDirectoryBase', type=str, help='')
    parser.add_argument('-d', '--delimiter',
                        metavar='delimiter', type=str, help='', default=";")
    parser.add_argument('-i', '--impalaHost',
                        metavar='impalaHost', type=str, help='')
    parser.add_argument('-o', '--impalaPort',
                        metavar='impalaPort', type=str, help='')
    parser.add_argument('-sl', '--solrServer',
                        metavar='solrServer', type=str, help='')
    parser.add_argument('-ju', '--jdbcUser',
                        metavar='jdbcUser', type=str, help='')
    parser.add_argument('-jp', '--jdbcPassword',
                        metavar='jdbcPassword', type=str, help='')
    parser.add_argument('-js', '--jdbcServer',
                        metavar='jdbcServer', type=str, help='')
    parser.add_argument('-jd', '--jdbcDatabase',
                        metavar='jdbcDatabase', type=str, help='')
    parser.add_argument('-sp', '--schemaPostgres',
                        metavar='schemaPostgres', type=str, help='')

    args = parser.parse_args()

    execute_process(args)