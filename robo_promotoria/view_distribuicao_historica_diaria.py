import pyspark
from utils import _update_impala_table


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_view_distribuicao_historica_diario") \
        .enableHiveSupport() \
        .getOrCreate()

spark.sql("drop view if exists exadata_aux.vw_distribuicao_historica_diaria")

spark.sql("""
        create view exadata_aux.vw_distribuicao_historica_diaria as
        SELECT  *
        from exadata_aux.tb_distribuicao_diaria
        union all
        select *
        from exadata_aux.tb_distribuicao_historica
""")

_update_impala_table("exadata_aux.vw_distribuicao_historica_diaria")
