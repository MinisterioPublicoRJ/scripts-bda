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

    """
    Regras:
    Denuncias e Cautelares - Soma simples de tudo que aparece em dado periodo.
    Acordos e Arquivamentos - Apenas considera 1. Por exemplo, caso haja mais de um
      acordo em um documento, contar apenas 1 vez. Ademais, se o ultimo movimento de
      acordo for de rescisao (ou desarquivamento no caso de arquivamentos) a soma fica zerada.
    """
    ANDAMENTOS_FILTERED = spark.sql("""
        SELECT *
        FROM {0}.mcpr_andamento s
        WHERE s.year_month >= cast(date_format(date_sub(current_timestamp(), 60), 'yyyyMM') as INT)
        and pcao_dt_andamento > cast(date_sub(current_timestamp(), 60) as timestamp)
        and pcao_dt_andamento <= current_timestamp()
    """.format(schema_exadata))
    ANDAMENTOS_FILTERED.registerTempTable('ANDAMENTOS_FILTERED')

    DOC_ACORDOS = spark.sql("""
        SELECT
            A.docu_dk,
            A.docu_orgi_orga_dk_responsavel,
            C.pcao_dt_andamento,
            D.stao_tppr_dk
        FROM {0}.mcpr_documento A
        JOIN {0}.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
        JOIN ANDAMENTOS_FILTERED C
        ON C.pcao_vist_dk = B.vist_dk 
        JOIN (
            SELECT *
            FROM {0}.mcpr_sub_andamento
            WHERE stao_tppr_dk IN (7827,7914,7883,7868,6361,6362,6391,7922,7928,7915,7917,7920) --7920 rescisao do acordo
            ) D
            ON D.stao_pcao_dk = C.pcao_dk
        INNER JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p
            ON p.pip_codigo = A.docu_orgi_orga_dk_responsavel
    """.format(schema_exadata, schema_exadata_aux))
    DOC_ACORDOS.registerTempTable("DOC_ACORDOS")
    MAX_DT_ACORD = spark.sql("""
        SELECT
            docu_dk,
            MAX(pcao_dt_andamento) AS max_dt_acordo
        FROM DOC_ACORDOS
        GROUP BY docu_dk
    """)
    MAX_DT_ACORD.registerTempTable("MAX_DT_ACORD")
    ACORDOS_POR_DOC = spark.sql("""
        SELECT
            docu_dk,
            docu_orgi_orga_dk_responsavel,
            MAX(de_30_a_60_dias) as acordos_de_30_a_60_dias,
            MAX(de_0_a_30_dias) as acordos_de_0_a_30_dias 
        FROM (
            SELECT
            CASE WHEN pcao_dt_andamento <= cast(date_sub(current_timestamp(), 30) as timestamp)
                      AND stao_tppr_dk IN (7827,7914,7883,7868,6361,6362,6391,7922,7928,7915,7917)
                THEN 1 ELSE 0 END as de_30_a_60_dias,
            CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), 30) as timestamp)
                      AND stao_tppr_dk IN (7827,7914,7883,7868,6361,6362,6391,7922,7928,7915,7917)
                THEN 1 ELSE 0 END as de_0_a_30_dias,
            A.docu_dk, A.docu_orgi_orga_dk_responsavel, A.pcao_dt_andamento, A.stao_tppr_dk
            FROM DOC_ACORDOS A
            JOIN MAX_DT_ACORD MDT ON MDT.docu_dk = A.docu_dk AND MDT.max_dt_acordo = A.pcao_dt_andamento) t
        GROUP BY docu_dk, docu_orgi_orga_dk_responsavel
    """)
    ACORDOS_POR_DOC.registerTempTable("ACORDOS_POR_DOC")
    NR_ACORDOS = spark.sql("""
        SELECT
            docu_orgi_orga_dk_responsavel as orgao_id,
            SUM(acordos_de_30_a_60_dias) as nr_acordos_ultimos_60_dias,
            SUM(acordos_de_0_a_30_dias) as nr_acordos_ultimos_30_dias
        FROM ACORDOS_POR_DOC
        GROUP BY docu_orgi_orga_dk_responsavel
    """.format(schema_exadata))
    NR_ACORDOS.registerTempTable("NR_ACORDOS")

    DOC_ARQUIVAMENTOS = spark.sql("""
        SELECT
            A.docu_dk,
            A.docu_orgi_orga_dk_responsavel,
            C.pcao_dt_andamento,
            D.stao_tppr_dk
        FROM {0}.mcpr_documento A
        JOIN {0}.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
        JOIN ANDAMENTOS_FILTERED C
        ON C.pcao_vist_dk = B.vist_dk 
        JOIN (
            SELECT *
            FROM {0}.mcpr_sub_andamento
            WHERE stao_tppr_dk IN (
                6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,6346,6350,6359,6392,6017,6018,6020,7745, --arquivamentos
                6075,1028,6798,7245,6307,1027,7803,6003,7802,7801) --desarquivamentos
            ) D
            ON D.stao_pcao_dk = C.pcao_dk
        INNER JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p
            ON p.pip_codigo = A.docu_orgi_orga_dk_responsavel
    """.format(schema_exadata, schema_exadata_aux))
    DOC_ARQUIVAMENTOS.registerTempTable("DOC_ARQUIVAMENTOS")
    MAX_DT_ARQUIV = spark.sql("""
        SELECT
            docu_dk,
            MAX(pcao_dt_andamento) AS max_dt_arquivamento
        FROM DOC_ARQUIVAMENTOS
        GROUP BY docu_dk
    """)
    MAX_DT_ARQUIV.registerTempTable("MAX_DT_ARQUIV")
    ARQUIVAMENTOS_POR_DOC = spark.sql("""
        SELECT
            docu_dk,
            docu_orgi_orga_dk_responsavel,
            MAX(de_30_a_60_dias) as de_30_a_60_dias,
            MAX(de_0_a_30_dias) as de_0_a_30_dias 
        FROM (
            SELECT
            CASE WHEN pcao_dt_andamento <= cast(date_sub(current_timestamp(), 30) as timestamp)
                      AND stao_tppr_dk IN (6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,6346,6350,6359,6392,6017,6018,6020,7745)
                THEN 1 ELSE 0 END as de_30_a_60_dias,
            CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), 30) as timestamp)
                      AND stao_tppr_dk IN (6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,6346,6350,6359,6392,6017,6018,6020,7745)
                THEN 1 ELSE 0 END as de_0_a_30_dias,
            A.docu_dk, A.docu_orgi_orga_dk_responsavel, A.pcao_dt_andamento, A.stao_tppr_dk
            FROM DOC_ARQUIVAMENTOS A
            JOIN MAX_DT_ARQUIV MDT ON MDT.docu_dk = A.docu_dk AND MDT.max_dt_arquivamento = A.pcao_dt_andamento) t
        GROUP BY docu_dk, docu_orgi_orga_dk_responsavel
    """)
    ARQUIVAMENTOS_POR_DOC.registerTempTable("ARQUIVAMENTOS_POR_DOC")
    NR_ARQUIVAMENTOS = spark.sql("""
        SELECT
            docu_orgi_orga_dk_responsavel as orgao_id,
            SUM(de_30_a_60_dias) as nr_arquivamentos_ultimos_60_dias,
            SUM(de_0_a_30_dias) as nr_arquivamentos_ultimos_30_dias
        FROM ARQUIVAMENTOS_POR_DOC
        GROUP BY docu_orgi_orga_dk_responsavel
    """.format(schema_exadata))
    NR_ARQUIVAMENTOS.registerTempTable("NR_ARQUIVAMENTOS")

    NR_CAUTELARES = spark.sql("""
        SELECT 
            orgao_id,
            SUM(de_30_a_60_dias) as nr_cautelares_ultimos_60_dias,
            SUM(de_0_a_30_dias) as nr_cautelares_ultimos_30_dias
        FROM (
            SELECT 
                CASE WHEN pcao_dt_andamento <= cast(date_sub(current_timestamp(), 30) as timestamp) THEN 1 ELSE 0 END as de_30_a_60_dias,
                CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), 30) as timestamp) THEN 1 ELSE 0 END as de_0_a_30_dias,
                docu_orgi_orga_dk_responsavel as orgao_id
            FROM {0}.mcpr_documento A
            JOIN {0}.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
            JOIN ANDAMENTOS_FILTERED C 
            ON C.pcao_vist_dk = B.vist_dk 
            JOIN (
                SELECT *
                FROM {0}.mcpr_sub_andamento
                WHERE stao_tppr_dk IN (
                    6648,6649,6650,6651,6652,6653,6654,6038,6039,6040,6041,6042,6043,7815,7816,6620,6257,6258,7878,7877,6367,6368,6369,6370,1208,1030) --Cautelares
                    ) D
            ON D.stao_pcao_dk = C.pcao_dk) t
        INNER JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p ON p.pip_codigo = t.orgao_id
        GROUP BY orgao_id
    """.format(schema_exadata, schema_exadata_aux))
    NR_CAUTELARES.registerTempTable("NR_CAUTELARES")

    NR_DENUNCIAS = spark.sql("""
        SELECT 
            orgao_id,
            SUM(de_30_a_60_dias) as nr_denuncias_ultimos_60_dias,
            SUM(de_0_a_30_dias) as nr_denuncias_ultimos_30_dias
        FROM (
            SELECT 
                CASE WHEN pcao_dt_andamento <= cast(date_sub(current_timestamp(), 30) as timestamp) THEN 1 ELSE 0 END as de_30_a_60_dias,
                CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), 30) as timestamp) THEN 1 ELSE 0 END as de_0_a_30_dias,
                docu_orgi_orga_dk_responsavel as orgao_id
            FROM {0}.mcpr_documento A
            JOIN {0}.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
            JOIN ANDAMENTOS_FILTERED C 
            ON C.pcao_vist_dk = B.vist_dk 
            JOIN (
                SELECT *
                FROM {0}.mcpr_sub_andamento
                WHERE stao_tppr_dk IN (6252,6253,1201,1202,6254) --Denuncias
                    ) D
            ON D.stao_pcao_dk = C.pcao_dk) t
        INNER JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p ON p.pip_codigo = t.orgao_id
        GROUP BY orgao_id
    """.format(schema_exadata, schema_exadata_aux))
    NR_DENUNCIAS.registerTempTable("NR_DENUNCIAS")

    table = spark.sql("""
        SELECT orgao_id, nm_orgao,
            nr_aproveitamentos_ultimos_60_dias,
            nr_aproveitamentos_ultimos_30_dias,
            CASE
                WHEN (nr_aproveitamentos_ultimos_30_dias - nr_aproveitamentos_ultimos_60_dias) = 0 THEN 0
                ELSE (nr_aproveitamentos_ultimos_30_dias - nr_aproveitamentos_ultimos_60_dias)/nr_aproveitamentos_ultimos_60_dias
            END as variacao_1_mes,
            nr_denuncias_ultimos_60_dias,
            nr_cautelares_ultimos_60_dias,
            nr_acordos_ultimos_60_dias,
            nr_arquivamentos_ultimos_60_dias,
            nr_denuncias_ultimos_30_dias,
            nr_cautelares_ultimos_30_dias,
            nr_acordos_ultimos_30_dias,
            nr_arquivamentos_ultimos_30_dias
        FROM (
            SELECT 
                p.pip_codigo as orgao_id, O.orgi_nm_orgao as nm_orgao,
                nvl(nr_denuncias_ultimos_60_dias, 0) as nr_denuncias_ultimos_60_dias,
                nvl(nr_cautelares_ultimos_60_dias, 0) as nr_cautelares_ultimos_60_dias,
                nvl(nr_acordos_ultimos_60_dias, 0) as nr_acordos_ultimos_60_dias,
                nvl(nr_arquivamentos_ultimos_60_dias, 0) as nr_arquivamentos_ultimos_60_dias,
                nvl(nr_denuncias_ultimos_30_dias, 0) as nr_denuncias_ultimos_30_dias,
                nvl(nr_cautelares_ultimos_30_dias, 0) as nr_cautelares_ultimos_30_dias,
                nvl(nr_acordos_ultimos_30_dias, 0) as nr_acordos_ultimos_30_dias,
                nvl(nr_arquivamentos_ultimos_30_dias, 0) as nr_arquivamentos_ultimos_30_dias,
                nvl(nr_denuncias_ultimos_60_dias, 0) + nvl(nr_cautelares_ultimos_60_dias, 0) + 
                nvl(nr_acordos_ultimos_60_dias, 0) + nvl(nr_arquivamentos_ultimos_60_dias, 0) AS nr_aproveitamentos_ultimos_60_dias,
                nvl(nr_denuncias_ultimos_30_dias, 0) + nvl(nr_cautelares_ultimos_30_dias, 0) + 
                nvl(nr_acordos_ultimos_30_dias, 0) + nvl(nr_arquivamentos_ultimos_30_dias, 0) AS nr_aproveitamentos_ultimos_30_dias
            FROM (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p
            JOIN {0}.orgi_orgao O ON orgi_dk = p.pip_codigo
            LEFT JOIN NR_DENUNCIAS A ON p.pip_codigo = A.orgao_id
            LEFT JOIN NR_CAUTELARES B ON p.pip_codigo= B.orgao_id
            LEFT JOIN NR_ACORDOS C ON p.pip_codigo = C.orgao_id
            LEFT JOIN NR_ARQUIVAMENTOS D ON p.pip_codigo = D.orgao_id) t
    """.format(schema_exadata, schema_exadata_aux)
    ).withColumn(
        "dt_inclusao",
        from_unixtime(
            unix_timestamp(current_timestamp(), 'yyyy-MM-dd HH:mm:ss'), 'yyyy-MM-dd HH:mm:ss')
        .cast('timestamp')
    )

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
