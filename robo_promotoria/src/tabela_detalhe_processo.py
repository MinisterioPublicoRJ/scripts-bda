import argparse
import pyspark

from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from generic_utils import execute_compute_stats

def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_detalhe_processo") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']
    table_name = options['table_name']

    table = spark.sql("""
        SELECT orgao_id, orgi_nm_orgao as nm_orgao, cod_pct,
            SUM(de_12_a_24) as nr_acoes_12_meses_anterior,
            SUM(de_0_a_12) as nr_acoes_12_meses_atual,
            SUM(de_60_dias_anterior) as nr_acoes_60_dias_anterior,
            SUM(de_60_dias_atual) as nr_acoes_ultimos_60_dias,
            SUM(de_30_dias_anterior) as nr_acoes_30_dias_anterior,
            SUM(de_30_dias_atual) as nr_acoes_ultimos_30_dias,
            CASE 
                WHEN (SUM(de_0_a_12) - SUM(de_12_a_24)) = 0 THEN 0
                ELSE (SUM(de_0_a_12) - SUM(de_12_a_24))/SUM(de_12_a_24)
            END as variacao_12_meses,
            CASE
                WHEN (SUM(de_60_dias_atual) - SUM(de_60_dias_anterior)) = 0 THEN 0
                ELSE (SUM(de_60_dias_atual) - SUM(de_60_dias_anterior))/SUM(de_60_dias_anterior)
            END as variacao_60_dias,
            CASE
                WHEN (SUM(de_30_dias_atual) - SUM(de_30_dias_anterior)) = 0 THEN 0
                ELSE (SUM(de_30_dias_atual) - SUM(de_30_dias_anterior))/SUM(de_30_dias_anterior)
            END as variacao_30_dias
        FROM (
            SELECT 
                CASE WHEN elapsed >= 365 THEN 1 ELSE 0 END as de_12_a_24,
                CASE WHEN elapsed < 365 THEN 1 ELSE 0 END as de_0_a_12,
                CASE WHEN elapsed < 120 AND elapsed >= 60 THEN 1 ELSE 0 END as de_60_dias_anterior,
                CASE WHEN elapsed < 60 THEN 1 ELSE 0 END as de_60_dias_atual,
                CASE WHEN elapsed < 60 AND elapsed >= 30 THEN 1 ELSE 0 END as de_30_dias_anterior,
                CASE WHEN elapsed < 30 THEN 1 ELSE 0 END as de_30_dias_atual,
                vist_orgi_orga_dk as orgao_id
            FROM {0}.mcpr_documento A
            JOIN {0}.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
            JOIN (
                SELECT *, datediff(current_timestamp(), pcao_dt_andamento) as elapsed
                FROM {0}.mcpr_andamento
                WHERE to_date(pcao_dt_andamento) > to_date(date_sub(current_timestamp(), 730))
                AND to_date(pcao_dt_andamento) <= to_date(current_timestamp())
                AND pcao_dt_cancelamento IS NULL
                ) C 
            ON C.pcao_vist_dk = B.vist_dk 
            JOIN (
                SELECT *
                FROM {0}.mcpr_sub_andamento
                WHERE stao_tppr_dk = 6251) D
            ON D.stao_pcao_dk = C.pcao_dk
            WHERE A.docu_tpst_dk != 11) t
        INNER JOIN {1}.atualizacao_pj_pacote p ON p.id_orgao = t.orgao_id
        GROUP BY orgao_id, orgi_nm_orgao, cod_pct
    """.format(schema_exadata, schema_exadata_aux))

    table_name = "{}.{}".format(schema_exadata_aux, table_name)

    table.write.mode("overwrite").saveAsTable("temp_table_detalhe_processo")
    temp_table = spark.table("temp_table_detalhe_processo")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_detalhe_processo")

    execute_compute_stats(table_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table detalhe processo")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    parser.add_argument('-t','--tableName', metavar='tableName', type=str, help='')
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort,
                    'table_name' : args.tableName,
                }

    execute_process(options)