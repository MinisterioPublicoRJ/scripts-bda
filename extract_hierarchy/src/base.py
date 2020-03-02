#-*-coding:utf-8-*-
import pyspark
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType, DateType

from timer import Timer

with Timer():
    print('Creating Spark Session')
    spark = pyspark.sql.session.SparkSession\
        .builder\
        .appName("alertas_dominio")\
        .enableHiveSupport()\
        .getOrCreate()

    sc = spark.sparkContext