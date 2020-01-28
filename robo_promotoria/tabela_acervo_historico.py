import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp
from utils import _update_impala_table


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_acervo_historico") \
        .enableHiveSupport() \
        .getOrCreate()

table = spark.sql("""
        SELECT * from exadata_aux.tb_acervo_diario
""")

table.write.mode("append").format("parquet").saveAsTable("exadata_aux.tb_acervo_historico")

spark.sql("""
       TRUNCATE TABLE exadata_aux.tb_acervo_diario
""")

_update_impala_table("exadata_aux.tb_acervo_historico")
