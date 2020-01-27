import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_acervo_historico") \
        .enableHiveSupport() \
        .getOrCreate()

table = spark.sql("""
        SELECT * from exadata_aux.tb_acervo_diario
""")

table.withColumn("data", from_unixtime(unix_timestamp(current_timestamp(), 'yyyy-MM-dd'), 'yyyy-MM-dd').cast('timestamp')) \
    .write.mode("append").format("parquet").saveAsTable("exadata_aux.tb_acervo_historico")


