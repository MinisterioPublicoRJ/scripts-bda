import argparse

import pyspark
from hdfs import InsecureClient
from pyspark.sql.functions import year, col, regexp_replace
from utils import _update_impala_table, send_log, ERROR, SUCCESS, SUCCESS_MESSAGE, ERROR_MESSAGE


def remove_break_lines(df, (col_name, dtype)):
    
    df = df.withColumn(col_name, regexp_replace(col(col_name),'\r','').cast(dtype))

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

    client = InsecureClient(args.webHdfs, args.userWebHdfs)

    hdfs_files = client.list(args.pathDirectoryBase)

    for directory in hdfs_files:

        try:

            actual_directory = args.pathDirectoryBase + directory

            df = spark.read.text(actual_directory)

            if not df.rdd.isEmpty():

                # df = spark.read.load(actual_directory, format="csv", multiLine=True,
			    #                     sep=args.delimiter, inferSchema=True, header=True)
                
                df = spark.read.load(actual_directory, format="csv", sep=args.delimiter, 
                            inferSchema=True, header=True)

                columns = [column_name.replace(" ", "_").replace("\r", "") for column_name in df.columns]
                
                df = df.toDF(*columns)

                #df = reduce(remove_break_lines, df.dtypes, df)
                    
                table_hive = "{}.{}".format(args.schemaHive, directory)
                
                table_postgres = "{}.{}".format(args.schemaPostgres, directory)

                df.write.mode("overwrite").format("parquet").saveAsTable(table_hive)

                _update_impala_table(table_hive, args.impalaHost, args.impalaPort)

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