import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from utils import _update_impala_table
import argparse


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_detalhe_acervo") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    table = spark.sql("""
        WITH tb_acervo_orgao_pct as (
                    SELECT *
                    FROM {namespace}.tb_acervo ac
                    INNER JOIN (
                        SELECT cod_pct
                        FROM {namespace}.atualizacao_pj_pacote
                        WHERE id_orgao = :orgao_id
                        ) org
                    ON org.cod_pct = ac.cod_atribuicao)
                SELECT
                    tb_data_fim.cod_orgao,
                    pc.orgi_nm_orgao as nm_orgao,
                    tb_data_fim.acervo_fim,
                    tb_data_inicio.acervo_inicio,
                    (acervo_fim - acervo_inicio)/acervo_inicio as variacao
                FROM (
                    SELECT cod_orgao, SUM(acervo) as acervo_fim
                    FROM tb_acervo_orgao_pct acpc
                    INNER JOIN {namespace}.tb_regra_negocio_investigacao regras
                        ON regras.cod_atribuicao = acpc.cod_atribuicao
                        AND regras.classe_documento = acpc.tipo_acervo
                    WHERE dt_inclusao = to_timestamp(:dt_fim, 'yyyy-MM-dd')
                    GROUP BY cod_orgao
                    ) tb_data_fim
                INNER JOIN (
                    SELECT cod_orgao, SUM(acervo) as acervo_inicio
                    FROM tb_acervo_orgao_pct acpc
                    INNER JOIN {namespace}.tb_regra_negocio_investigacao regras
                        ON regras.cod_atribuicao = acpc.cod_atribuicao
                        AND regras.classe_documento = acpc.tipo_acervo
                    WHERE dt_inclusao = to_timestamp(:dt_inicio, 'yyyy-MM-dd')
                    GROUP BY cod_orgao
                    ) tb_data_inicio
                ON tb_data_fim.cod_orgao = tb_data_inicio.cod_orgao
                INNER JOIN {namespace}.atualizacao_pj_pacote pc
                ON pc.id_orgao = tb_data_fim.cod_orgao
    """.format(schema_exadata, schema_exadata_aux))

    table_name = "{}.tb_detalhe_processo".format(schema_exadata_aux)

    table.write.mode("overwrite").saveAsTable("temp_table_detalhe_processo")
    temp_table = spark.table("temp_table_detalhe_processo")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_detalhe_processo")

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table detalhe processo")
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
