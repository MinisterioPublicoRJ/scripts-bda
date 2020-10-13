import pyspark

def spark_session(app_name):
    return pyspark.sql.session.SparkSession\
        .builder\
        .appName(app_name)\
        .enableHiveSupport()\
        .getOrCreate()