import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp
from utils import _update_impala_table


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_view_acervo_historico_diario") \
        .enableHiveSupport() \
        .getOrCreate()

spark.sql("drop view if exists exadata_aux.vw_acervo_historico_diario")

spark.sql("""
        create view exadata_aux.vw_acervo_historico_diario as 
        SELECT  *
        from exadata_aux.tb_acervo_diario
        union all
        select * 
        from  exadata_aux.tb_acervo_historico
""")

_update_impala_table("exadata_aux.vw_acervo_historico_diario")

