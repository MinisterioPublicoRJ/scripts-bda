import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from utils import _update_impala_table


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_saida") \
        .enableHiveSupport() \
        .getOrCreate()

table = spark.sql("""
    SELECT  saidas, cast(id_orgao as int) as id_orgao, 
            cod_pct, percent_rank() over (PARTITION BY cod_pct order by saidas) as percent_rank, 
            current_timestamp() as dt_calculo
    FROM (
    SELECT COUNT(pcao_dt_andamento) as saidas, t2.id_orgao, t2.cod_pct
    FROM (
        SELECT F.id_orgao, C.pcao_dt_andamento
        FROM exadata_dev.mcpr_documento A
        JOIN exadata_dev.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
        JOIN exadata_dev.mcpr_andamento C on C.pcao_vist_dk = B.vist_dk AND C.pcao_dt_andamento >= to_date(date_sub(current_timestamp(), 60))
        JOIN exadata_dev.mcpr_sub_andamento D on D.stao_pcao_dk = C.pcao_dk
        JOIN cluster.atualizacao_pj_pacote F ON A.docu_orgi_orga_dk_responsavel = cast(F.id_orgao as int)
        JOIN exadata_aux.tb_regra_negocio_saida on cod_atribuicao = F.cod_pct and D.stao_tppr_dk = tp_andamento
        ) t1
    RIGHT JOIN (
            select * 
            from cluster.atualizacao_pj_pacote p
            where p.cod_pct in (SELECT DISTINCT cod_atribuicao FROM exadata_aux.tb_regra_negocio_saida)
        ) t2 on t2.id_orgao = t1.id_orgao
    GROUP BY t2.id_orgao, cod_pct) t
""")

table.write.mode("overwrite").saveAsTable("exadata_aux.tb_saida")

_update_impala_table("exadata_aux.tb_saida")