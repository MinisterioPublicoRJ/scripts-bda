import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp
from utils import _update_impala_table


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_view_acervo_historico_diario") \
        .enableHiveSupport() \
        .getOrCreate()

spark.sql("drop table if exists exadata_aux.vw_acervo_historico_diario")

spark.sql("""
        CREATE TABLE exadata_aux.vw_acervo_historico_diario as 
        SELECT  cod_orgao, 
                pacote_atribuicao, 
                acervo, 
                tipo_acervo, 
                dt_inclusao
        from exadata_aux.tb_acervo_diario
        union all
        select * 
        from  exadata_aux.tb_acervo_historico
""")

_update_impala_table("exadata_aux.vw_acervo_historico_diario")

