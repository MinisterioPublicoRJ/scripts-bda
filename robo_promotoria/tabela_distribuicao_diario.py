import pyspark

from pyspark.sql.functions import (
    unix_timestamp,
    from_unixtime,
    current_timestamp,
)

from utils import _update_impala_table


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_distribuicao_diaria") \
        .enableHiveSupport() \
        .getOrCreate()


estatisticas = spark.sql(
    """
    select pacote_atribuicao,
    min(acervo) as minino,
    max(acervo) as maximo,
    avg(acervo) as media,
    percentile(acervo, 0.25) as primeiro_quartil,
    percentile(acervo, 0.5) as mediana,
    percentile(acervo, 0.75) as terceiro_quartil,
    percentile(acervo, 0.75) - percentile(acervo, 0.25) as IQR,
    percentile(acervo, 0.25)
        - 1.5*(percentile(acervo, 0.75) - percentile(acervo, 0.25)) as Lout,
    percentile(acervo, 0.75)
        + 1.5*(percentile(acervo, 0.75) - percentile(acervo, 0.25)) as Hout
    from exadata_aux.tb_acervo_diario group by pacote_atribuicao
    """
).withColumn(
    "data",
    from_unixtime(
        unix_timestamp(current_timestamp(), 'yyyy-MM-dd'), 'yyyy-MM-dd')
    .cast('timestamp')
).write.mode("overwrite").format("parquet")\
    .saveAsTable("exadata_aux.tb_distribuicao_diaria")

_update_impala_table("exadata_aux.tb_distribuicao_diaria")
