#-*-coding:utf-8-*-
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import *

from decouple import config
from base import spark

schema_exadata = config('SCHEMA_EXADATA')
schema_exadata_aux = config('SCHEMA_EXADATA_AUX')

def class_hierarchy():
    classe = spark.table('%s.mcpr_documento' % schema_exadata)
