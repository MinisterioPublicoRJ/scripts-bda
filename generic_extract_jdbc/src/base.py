import pyspark
from timer import timer
import os


READONLYDB = 'readonlydb'
BASES = 'bases' + os.environ.get('SUFIXO_BASES', '_dev')
DADOSSINAPSE = 'dadossinapse' + os.environ.get('SUFIXO_DADOSSINAPSE', '_dev')
DESTFOLDER = 'dadossinapse' + os.environ.get('SUFIXO_DESTFOLDER', '_dev')


with timer():
    print('Creating Spark Session')
    spark = pyspark.sql.session.SparkSession\
        .builder\
        .appName("GENERIC_LOAD_TABLE")\
        .enableHiveSupport()\
        .getOrCreate()

    sc = spark.sparkContext

    spark.sql("SET mapreduce.input.fileinputformat.input.dir.recursive=true")


    spark.sql('create database if not exists %s' % BASES)
    spark.sql('create database if not exists %s' % DADOSSINAPSE)
