import pyspark
from datetime import datetime

from pyspark.sql.functions import (
    unix_timestamp,
    from_unixtime,
    current_timestamp,
    date_format
)

from utils import _update_impala_table


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_distribuicao") \
        .enableHiveSupport() \
        .getOrCreate()

spark.conf.set("spark.sql.sources.partitionOverwriteMode","dynamic")

spark.sql("use %s" % "exadata_aux")
result_table_check = spark.sql("SHOW TABLES LIKE '%s'" % "tb_distribuicao").count()

date_now = datetime.now()
data_atual = date_now.strftime("%Y-%m-%d")

estatisticas = spark.sql(
    """
    select pacote_atribuicao,
    tipo_acervo,
    min(acervo) as minimo,
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
    from exadata_aux.tb_acervo 
    where dt_inclusao = '{}'
    group by pacote_atribuicao, tipo_acervo
    """.format(data_atual)
).withColumn(
    "dt_inclusao",
    from_unixtime(
        unix_timestamp(current_timestamp(), 'yyyy-MM-dd'), 'yyyy-MM-dd')
    .cast('timestamp')
).withColumn("dt_partition", date_format(current_timestamp(), "ddMMyyyy"))


if result_table_check > 0:
    estatisticas.write.mode("overwrite").insertInto("exadata_aux.tb_distribuicao", overwrite=True)
else:
    estatisticas.write.partitionBy("dt_partition").saveAsTable("exadata_aux.tb_distribuicao")

_update_impala_table("exadata_aux.tb_distribuicao")
