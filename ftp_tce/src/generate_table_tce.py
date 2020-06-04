import argparse

import pyspark
from hdfs import InsecureClient
from pyspark.sql.functions import year
from utils import _update_impala_table, send_log, ERROR, SUCCESS, SUCCESS_MESSAGE, ERROR_MESSAGE
#from send_request_solr import send_log, ERROR, SUCCESS, SUCCESS_MESSAGE, ERROR_MESSAGE


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

            schema_tce = args.schemaTce

            df = spark.read.text(actual_directory)

            if not df.rdd.isEmpty():

                df = spark.read.option("encoding", "ISO-8859-1").load(actual_directory, format="csv",
                                    sep=args.delimiter, inferSchema=True, header=True)
                
                columns = [column_name.replace(" ", "_") for column_name in df.columns]
                df = df.toDF(*columns)
                    
                table = "{}.{}".format(schema_tce, directory)
                
                df.write.mode("overwrite").format("parquet").saveAsTable(table)

                _update_impala_table(table, args.impalaHost, args.impalaPort)

                send_log(SUCCESS_MESSAGE.format(directory), app_name, SUCCESS, args.solrServer)

        except Exception as message:
            send_log(ERROR_MESSAGE.format(directory, message), app_name, ERROR, args.solrServer)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="criar tabela tce")
    parser.add_argument('-st', '--schemaTce',
                        metavar='schemaTce', type=str, help='')
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
    args = parser.parse_args()

    execute_process(args)

