import pyspark
from utils import _update_impala_table


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_view_distribuicao_historica_diario") \
        .enableHiveSupport() \
        .getOrCreate()

spark.sql("drop table if exists exadata_aux.vw_distribuicao_historica_diaria")

spark.sql("""
        create table exadata_aux.vw_distribuicao_historica_diaria as
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
        union all
        select *
        from exadata_aux.tb_distribuicao_historica
""")

_update_impala_table("exadata_aux.vw_distribuicao_historica_diaria")
