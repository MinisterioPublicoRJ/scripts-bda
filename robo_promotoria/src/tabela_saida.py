import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
import argparse
from generic_utils import execute_compute_stats


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_saida") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    table = spark.sql("""
        SELECT  saidas, cast(id_orgao as int) as id_orgao, 
                cod_pct, percent_rank() over (PARTITION BY cod_pct order by saidas) as percent_rank, 
                current_timestamp() as dt_calculo
        FROM (
        SELECT COUNT(pcao_dt_andamento) as saidas, t2.id_orgao, t2.cod_pct
        FROM (
            SELECT F.id_orgao, C.pcao_dt_andamento
            FROM {0}.mcpr_documento A
            JOIN {0}.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
            JOIN {0}.mcpr_andamento C on C.pcao_vist_dk = B.vist_dk
            JOIN {0}.mcpr_sub_andamento D on D.stao_pcao_dk = C.pcao_dk
            JOIN {1}.atualizacao_pj_pacote F ON B.vist_orgi_orga_dk = cast(F.id_orgao as int)
            JOIN {1}.tb_regra_negocio_saida on cod_atribuicao = F.cod_pct and D.stao_tppr_dk = tp_andamento
            WHERE C.pcao_dt_cancelamento IS NULL
            AND A.docu_tpst_dk != 11
	    AND C.year_month >= cast(date_format(date_sub(current_timestamp(), 180), 'yyyyMM') as INT)
            AND C.pcao_dt_andamento >= date_sub(current_timestamp(), 180)
            ) t1
        RIGHT JOIN (
                select * 
                from {1}.atualizacao_pj_pacote p
                where p.cod_pct in (SELECT DISTINCT cod_atribuicao FROM {1}.tb_regra_negocio_saida)
            ) t2 on t2.id_orgao = t1.id_orgao
        GROUP BY t2.id_orgao, cod_pct) t
    """.format(schema_exadata, schema_exadata_aux))

    table_name = options['table_name']
    table_name = "{}.{}".format(schema_exadata_aux, table_name)

    table.write.mode("overwrite").saveAsTable("temp_table_saida")
    temp_table = spark.table("temp_table_saida")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_saida")

    execute_compute_stats(table_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table tabela saida")
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
