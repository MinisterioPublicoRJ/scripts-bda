import pyspark
from utils import _update_impala_table


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_distribuicao_historica")\
        .enableHiveSupport() \
        .getOrCreate()

spark.sql("""
        SELECT  pacote_atribuicao,
                minino,
                maximo,
                media,
                primeiro_quartil,
                mediana,
                terceiro_quartil,
                iqr,
                lout,
                hout,
                dt_inclusao
        from exadata_aux.tb_distribuicao_diaria
""").write.mode("append").format("parquet").saveAsTable(
    "exadata_aux.tb_distribuicao_historica"
)

spark.sql(" DROP TABLE exadata_aux.tb_distribuicao_diaria ")

_update_impala_table("exadata_aux.tb_distribuicao_historica")
