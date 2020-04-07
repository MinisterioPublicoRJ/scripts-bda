import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from utils import _update_impala_table
import argparse


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_pip_detalhe_aproveitamentos") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    table = spark.sql("""
        SELECT orgao_id, orgi_nm_orgao as nm_orgao,
            SUM(de_30_a_60_dias) as nr_aproveitamentos_ultimos_60_dias,
            SUM(de_0_a_30_dias) as nr_aproveitamentos_ultimos_30_dias,
            CASE
            WHEN (SUM(de_0_a_30_dias) - SUM(de_30_a_60_dias)) = 0 THEN 0
            ELSE (SUM(de_0_a_30_dias) - SUM(de_30_a_60_dias))/SUM(de_30_a_60_dias)
            END as variacao_1_mes
        FROM (
            SELECT 
                CASE WHEN to_date(pcao_dt_andamento) <= to_date(date_sub(current_timestamp(), 30)) THEN 1 ELSE 0 END as de_30_a_60_dias,
                CASE WHEN to_date(pcao_dt_andamento) > to_date(date_sub(current_timestamp(), 30)) THEN 1 ELSE 0 END as de_0_a_30_dias,
                docu_orgi_orga_dk_responsavel as orgao_id
            FROM {0}.mcpr_documento A
            JOIN {0}.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
            JOIN (
                SELECT *
                FROM {0}.mcpr_andamento
                WHERE to_date(pcao_dt_andamento) > to_date(date_sub(current_timestamp(), 60))
                AND to_date(pcao_dt_andamento) <= to_date(current_timestamp())) C 
            ON C.pcao_vist_dk = B.vist_dk 
            JOIN (
                SELECT *
                FROM {0}.mcpr_sub_andamento
                WHERE stao_tppr_dk IN (
                    6648,6649,6650,6651,6652,6653,6654,6038,6039,6040,6041,6042,6043,7815,7816,6620,6257,6258,7878,7877,6367,6368,6369,6370,1208,1030) --Cautelares
                    ) D
            ON D.stao_pcao_dk = C.pcao_dk) t
        INNER JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p ON p.pip_codigo = t.orgao_id
        INNER JOIN exadata.orgi_orgao ON orgi_dk = t.orgao_id
        GROUP BY orgao_id, orgi_nm_orgao
    """.format(schema_exadata, schema_exadata_aux))

    table_name = "{}.tb_pip_detalhe_aproveitamentos".format(schema_exadata_aux)

    table.write.mode("overwrite").saveAsTable("temp_table_pip_detalhe_aproveitamentos")
    temp_table = spark.table("temp_table_pip_detalhe_aproveitamentos")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_pip_detalhe_aproveitamentos")

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table pip detalhe aproveitamentos")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort
                }

    execute_process(options)