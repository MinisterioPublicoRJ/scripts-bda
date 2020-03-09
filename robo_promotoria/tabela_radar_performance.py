import pyspark
from utils import _update_impala_table
from decouple import config


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_radar_performance") \
        .enableHiveSupport() \
        .getOrCreate()

schema_exadata = config('SCHEMA_EXADATA')
schema_exadata_aux = config('SCHEMA_EXADATA_AUX')


table = spark.sql("""
    SELECT
        orgao_id,
        SUM(is_arquivamento) as nr_arquivamentos,
        SUM(is_indeferimento) as nr_indeferimentos,
        SUM(is_instauracao) as nr_instauracao,
        SUM(is_tac) as nr_tac,
        SUM(is_acao) as nr_acoes,
        to_date(current_timestamp()) as dt_calculo
    FROM (
        SELECT
            docu_orgi_orga_dk_responsavel as orgao_id,
            CASE WHEN stao_tppr_dk IN (7912,6548,6326,6681,6678,6645,6682,6680,6679,6644,
                6668,6666,6665,6669,6667,6664,6655,6662,6659,6658,6663,6661,6660,6657,
                6670,6676,6674,6673,6677,6675,6672,6018,6341,6338,6019,6017,6591,6339,
                6553,7871,6343,6340,6342,6021,6334,6331,6022,6020,6593,6332,7872,6336,
                6333,6335,7745,6346,6345,
                6015, 6016, 6325, 6327, 6328, 6329, 6330, 6337, 6344, 6656, 6671, 7869, 7870, 6324) THEN 1 ELSE 0 END as is_arquivamento,
            CASE WHEN stao_tppr_dk = 6322 THEN 1 WHEN stao_tppr_dk = 6007 THEN -1 ELSE 0 END as is_indeferimento,
            CASE WHEN stao_tppr_dk IN (6011, 6012, 6013, 1092, 1094, 1095) THEN 1 ELSE 0 END as is_instauracao,
            CASE WHEN stao_tppr_dk IN (6655, 6326, 6370) THEN 1 ELSE 0 END as is_tac,
            CASE WHEN stao_tppr_dk = 6251 THEN 1 ELSE 0 END as is_acao
        FROM {0}.mcpr_documento A
        JOIN {0}.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
        JOIN (
            SELECT *
            FROM {0}.mcpr_andamento
            WHERE to_date(pcao_dt_andamento) > to_date(date_sub(current_timestamp(), 365))
            AND to_date(pcao_dt_andamento) <= to_date(current_timestamp())) C
        ON C.pcao_vist_dk = B.vist_dk
        JOIN {0}.mcpr_sub_andamento D ON D.stao_pcao_dk = C.pcao_dk
        ) t
    GROUP BY orgao_id
""".format(schema_exadata, schema_exadata_aux))

table_name = "{}.tb_radar_performance".format(schema_exadata_aux)

table.write.mode("overwrite").saveAsTable("temp_table_radar_performance")
temp_table = spark.table("temp_table_radar_performance")

temp_table.write.mode("overwrite").saveAsTable(table_name)
spark.sql("drop table temp_table_radar_performance")

_update_impala_table(table_name)
