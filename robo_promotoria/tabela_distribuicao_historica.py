import pyspark
from utils import _update_impala_table


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_distribuicao_historica")\
        .enableHiveSupport() \
        .getOrCreate()

spark.sql("""
        SELECT * from exadata_aux.tb_distribuicao_diaria
""").write.mode("append").format("parquet").saveAsTable(
    "exadata_aux.tb_distribuicao_historica"
)
_update_impala_table("exadata_aux.tb_distribuicao_historica")
