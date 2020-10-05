import argparse
import pyspark

from generic_utils import execute_compute_stats
from pyspark.sql.functions import (
    unix_timestamp, 
    from_unixtime, 
    current_timestamp, 
    lit, 
    date_format
)


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_pip_detalhe_aproveitamentos") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    tamanho_periodo_dias = options['nb_past_days']

    output_table_name = options['table_name']

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
        WHERE s.year_month >= cast(date_format(date_sub(current_timestamp(), {1}), 'yyyyMM') as INT)
        and pcao_dt_andamento > cast(date_sub(current_timestamp(), {1}) as timestamp)
        and pcao_dt_andamento <= current_timestamp()
        and pcao_dt_cancelamento IS NULL
    """.format(schema_exadata, 2*tamanho_periodo_dias))
    ANDAMENTOS_FILTERED.registerTempTable('ANDAMENTOS_FILTERED')

    # Numero de acordos
    DOC_ACORDOS = spark.sql("""
        SELECT
            A.docu_dk,
            B.vist_orgi_orga_dk,
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
            ON p.pip_codigo = B.vist_orgi_orga_dk
        WHERE docu_cldc_dk IN (3, 494, 590)
        AND docu_tpst_dk != 11
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
            vist_orgi_orga_dk,
            MAX(periodo_anterior) as acordos_periodo_anterior,
            MAX(periodo_atual) as acordos_periodo_atual 
        FROM (
            SELECT
            CASE WHEN pcao_dt_andamento <= cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (7827,7914,7883,7868,6361,6362,6391,7922,7928,7915,7917)
                THEN 1 ELSE 0 END as periodo_anterior,
            CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (7827,7914,7883,7868,6361,6362,6391,7922,7928,7915,7917)
                THEN 1 ELSE 0 END as periodo_atual,
            A.docu_dk, A.vist_orgi_orga_dk, A.pcao_dt_andamento, A.stao_tppr_dk
            FROM DOC_ACORDOS A
            JOIN MAX_DT_ACORD MDT ON MDT.docu_dk = A.docu_dk AND MDT.max_dt_acordo = A.pcao_dt_andamento) t
        GROUP BY docu_dk, vist_orgi_orga_dk
    """.format(tamanho_periodo_dias))
    ACORDOS_POR_DOC.registerTempTable("ACORDOS_POR_DOC")
    NR_ACORDOS = spark.sql("""
        SELECT
            vist_orgi_orga_dk as orgao_id,
            SUM(acordos_periodo_anterior) as nr_acordos_periodo_anterior,
            SUM(acordos_periodo_atual) as nr_acordos_periodo_atual
        FROM ACORDOS_POR_DOC
        GROUP BY vist_orgi_orga_dk
    """.format(schema_exadata))
    NR_ACORDOS.registerTempTable("NR_ACORDOS")

    # Numero de acordos de nao persecucao
    DOC_ACORDOS_N_PERSECUCAO = spark.sql("""
        SELECT
            A.docu_dk,
            B.vist_orgi_orga_dk,
            C.pcao_dt_andamento,
            D.stao_tppr_dk
        FROM {0}.mcpr_documento A
        JOIN {0}.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
        JOIN ANDAMENTOS_FILTERED C
        ON C.pcao_vist_dk = B.vist_dk 
        JOIN (
            SELECT *
            FROM {0}.mcpr_sub_andamento
            WHERE stao_tppr_dk IN (7914,7917,7928,7883,7915,7922,7827,7920) --7920 rescisao do acordo
            ) D
            ON D.stao_pcao_dk = C.pcao_dk
        INNER JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p
            ON p.pip_codigo = B.vist_orgi_orga_dk
        WHERE docu_cldc_dk IN (3, 494, 590)
        AND docu_tpst_dk != 11
    """.format(schema_exadata, schema_exadata_aux))
    DOC_ACORDOS_N_PERSECUCAO.registerTempTable("DOC_ACORDOS_N_PERSECUCAO")
    MAX_DT_ACORD_N_PERSECUCAO = spark.sql("""
        SELECT
            docu_dk,
            MAX(pcao_dt_andamento) AS max_dt_acordo
        FROM DOC_ACORDOS_N_PERSECUCAO
        GROUP BY docu_dk
    """)
    MAX_DT_ACORD_N_PERSECUCAO.registerTempTable("MAX_DT_ACORD_N_PERSECUCAO")
    ACORDOS_POR_DOC_N_PERSECUCAO = spark.sql("""
        SELECT
            docu_dk,
            vist_orgi_orga_dk,
            MAX(periodo_anterior) as acordos_periodo_anterior,
            MAX(periodo_atual) as acordos_periodo_atual 
        FROM (
            SELECT
            CASE WHEN pcao_dt_andamento <= cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (7914,7917,7928,7883,7915,7922,7827)
                THEN 1 ELSE 0 END as periodo_anterior,
            CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (7914,7917,7928,7883,7915,7922,7827)
                THEN 1 ELSE 0 END as periodo_atual,
            A.docu_dk, A.vist_orgi_orga_dk, A.pcao_dt_andamento, A.stao_tppr_dk
            FROM DOC_ACORDOS_N_PERSECUCAO A
            JOIN MAX_DT_ACORD_N_PERSECUCAO MDT ON MDT.docu_dk = A.docu_dk AND MDT.max_dt_acordo = A.pcao_dt_andamento) t
        GROUP BY docu_dk, vist_orgi_orga_dk
    """.format(tamanho_periodo_dias))
    ACORDOS_POR_DOC_N_PERSECUCAO.registerTempTable("ACORDOS_POR_DOC_N_PERSECUCAO")
    NR_ACORDOS_N_PERSECUCAO = spark.sql("""
        SELECT
            vist_orgi_orga_dk as orgao_id,
            SUM(acordos_periodo_anterior) as nr_acordos_n_persecucao_periodo_anterior,
            SUM(acordos_periodo_atual) as nr_acordos_n_persecucao_periodo_atual
        FROM ACORDOS_POR_DOC_N_PERSECUCAO
        GROUP BY vist_orgi_orga_dk
    """.format(schema_exadata))
    NR_ACORDOS_N_PERSECUCAO.registerTempTable("NR_ACORDOS_N_PERSECUCAO")

    # Numero de Arquivamentos
    DOC_ARQUIVAMENTOS = spark.sql("""
        SELECT
            A.docu_dk,
            B.vist_orgi_orga_dk,
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
            ON p.pip_codigo = B.vist_orgi_orga_dk
        WHERE docu_cldc_dk IN (3, 494, 590)
        AND docu_tpst_dk != 11
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
            vist_orgi_orga_dk,
            MAX(periodo_anterior) as periodo_anterior,
            MAX(periodo_atual) as periodo_atual 
        FROM (
            SELECT
            CASE WHEN pcao_dt_andamento <= cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,6346,6350,6359,6392,6017,6018,6020,7745)
                THEN 1 ELSE 0 END as periodo_anterior,
            CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,6346,6350,6359,6392,6017,6018,6020,7745)
                THEN 1 ELSE 0 END as periodo_atual,
            A.docu_dk, A.vist_orgi_orga_dk, A.pcao_dt_andamento, A.stao_tppr_dk
            FROM DOC_ARQUIVAMENTOS A
            JOIN MAX_DT_ARQUIV MDT ON MDT.docu_dk = A.docu_dk AND MDT.max_dt_arquivamento = A.pcao_dt_andamento) t
        GROUP BY docu_dk, vist_orgi_orga_dk
    """.format(tamanho_periodo_dias))
    ARQUIVAMENTOS_POR_DOC.registerTempTable("ARQUIVAMENTOS_POR_DOC")
    NR_ARQUIVAMENTOS = spark.sql("""
        SELECT
            vist_orgi_orga_dk as orgao_id,
            SUM(periodo_anterior) as nr_arquivamentos_periodo_anterior,
            SUM(periodo_atual) as nr_arquivamentos_periodo_atual
        FROM ARQUIVAMENTOS_POR_DOC
        GROUP BY vist_orgi_orga_dk
    """.format(schema_exadata))
    NR_ARQUIVAMENTOS.registerTempTable("NR_ARQUIVAMENTOS")

    # Numero de medidas cautelares
    NR_CAUTELARES = spark.sql("""
        SELECT 
            orgao_id,
            SUM(periodo_anterior) as nr_cautelares_periodo_anterior,
            SUM(periodo_atual) as nr_cautelares_periodo_atual
        FROM (
            SELECT 
                CASE WHEN pcao_dt_andamento <= cast(date_sub(current_timestamp(), {2}) as timestamp) THEN 1 ELSE 0 END as periodo_anterior,
                CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), {2}) as timestamp) THEN 1 ELSE 0 END as periodo_atual,
                vist_orgi_orga_dk as orgao_id
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
            ON D.stao_pcao_dk = C.pcao_dk
            WHERE docu_cldc_dk IN (3, 494, 590)
            AND docu_tpst_dk != 11
        ) t
        INNER JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p ON p.pip_codigo = t.orgao_id
        GROUP BY orgao_id
    """.format(schema_exadata, schema_exadata_aux, tamanho_periodo_dias))
    NR_CAUTELARES.registerTempTable("NR_CAUTELARES")

    # Numero de denuncias
    NR_DENUNCIAS = spark.sql("""
        SELECT 
            orgao_id,
            SUM(periodo_anterior) as nr_denuncias_periodo_anterior,
            SUM(periodo_atual) as nr_denuncias_periodo_atual
        FROM (
            SELECT 
                CASE WHEN pcao_dt_andamento <= cast(date_sub(current_timestamp(), {2}) as timestamp) THEN 1 ELSE 0 END as periodo_anterior,
                CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), {2}) as timestamp) THEN 1 ELSE 0 END as periodo_atual,
                vist_orgi_orga_dk as orgao_id
            FROM {0}.mcpr_documento A
            JOIN {0}.mcpr_vista B on B.vist_docu_dk = A.DOCU_DK
            JOIN ANDAMENTOS_FILTERED C 
            ON C.pcao_vist_dk = B.vist_dk 
            JOIN (
                SELECT *
                FROM {0}.mcpr_sub_andamento
                WHERE stao_tppr_dk IN (6252,6253,1201,1202,6254) --Denuncias
                    ) D
            ON D.stao_pcao_dk = C.pcao_dk
            WHERE docu_cldc_dk IN (3, 494, 590)
            AND docu_tpst_dk != 11
        ) t
        INNER JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p ON p.pip_codigo = t.orgao_id
        GROUP BY orgao_id
    """.format(schema_exadata, schema_exadata_aux, tamanho_periodo_dias))
    NR_DENUNCIAS.registerTempTable("NR_DENUNCIAS")

    # Numero de vistas abertas
    NR_VISTAS_ABERTAS = spark.sql("""
        SELECT
            vist_orgi_orga_dk as orgao_id,
            COUNT(vist_dk) as nr_aberturas_vista
        FROM {0}.mcpr_documento
        JOIN {0}.mcpr_vista ON vist_docu_dk = docu_dk
        INNER JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p 
            ON p.pip_codigo = vist_orgi_orga_dk
        WHERE docu_cldc_dk IN (3, 494, 590) -- PIC e Inqueritos
        AND vist_dt_abertura_vista >= cast(date_sub(current_timestamp(), {2}) as timestamp)
        AND docu_tpst_dk != 11
        GROUP BY vist_orgi_orga_dk
    """.format(schema_exadata, schema_exadata_aux, tamanho_periodo_dias))
    NR_VISTAS_ABERTAS.registerTempTable("NR_VISTAS_ABERTAS")

    table = spark.sql("""
        SELECT orgao_id, nm_orgao,
            nr_aproveitamentos_periodo_anterior,
            nr_aproveitamentos_periodo_atual,
            CASE
                WHEN (nr_aproveitamentos_periodo_atual - nr_aproveitamentos_periodo_anterior) = 0 THEN 0
                ELSE (nr_aproveitamentos_periodo_atual - nr_aproveitamentos_periodo_anterior)/nr_aproveitamentos_periodo_anterior
            END as variacao_periodo,
            nr_denuncias_periodo_anterior,
            nr_cautelares_periodo_anterior,
            nr_acordos_periodo_anterior,
            nr_acordos_n_persecucao_periodo_anterior,
            nr_arquivamentos_periodo_anterior,
            nr_denuncias_periodo_atual,
            nr_cautelares_periodo_atual,
            nr_acordos_periodo_atual,
            nr_acordos_n_persecucao_periodo_atual,
            nr_arquivamentos_periodo_atual,
            nr_aberturas_vista_periodo_atual
        FROM (
            SELECT 
                p.pip_codigo as orgao_id, O.orgi_nm_orgao as nm_orgao,
                nvl(nr_denuncias_periodo_anterior, 0) as nr_denuncias_periodo_anterior,
                nvl(nr_cautelares_periodo_anterior, 0) as nr_cautelares_periodo_anterior,
                nvl(nr_acordos_periodo_anterior, 0) as nr_acordos_periodo_anterior,
                nvl(nr_acordos_n_persecucao_periodo_anterior, 0) as nr_acordos_n_persecucao_periodo_anterior,
                nvl(nr_arquivamentos_periodo_anterior, 0) as nr_arquivamentos_periodo_anterior,
                nvl(nr_denuncias_periodo_atual, 0) as nr_denuncias_periodo_atual,
                nvl(nr_cautelares_periodo_atual, 0) as nr_cautelares_periodo_atual,
                nvl(nr_acordos_periodo_atual, 0) as nr_acordos_periodo_atual,
                nvl(nr_acordos_n_persecucao_periodo_atual, 0) as nr_acordos_n_persecucao_periodo_atual,
                nvl(nr_arquivamentos_periodo_atual, 0) as nr_arquivamentos_periodo_atual,
                nvl(nr_aberturas_vista, 0) as nr_aberturas_vista_periodo_atual,
                nvl(nr_denuncias_periodo_anterior, 0) + nvl(nr_cautelares_periodo_anterior, 0) + 
                nvl(nr_acordos_periodo_anterior, 0) + nvl(nr_arquivamentos_periodo_anterior, 0) AS nr_aproveitamentos_periodo_anterior,
                nvl(nr_denuncias_periodo_atual, 0) + nvl(nr_cautelares_periodo_atual, 0) + 
                nvl(nr_acordos_periodo_atual, 0) + nvl(nr_arquivamentos_periodo_atual, 0) AS nr_aproveitamentos_periodo_atual
            FROM (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) p
            JOIN {0}.orgi_orgao O ON orgi_dk = p.pip_codigo
            LEFT JOIN NR_DENUNCIAS A ON p.pip_codigo = A.orgao_id
            LEFT JOIN NR_CAUTELARES B ON p.pip_codigo= B.orgao_id
            LEFT JOIN NR_ACORDOS C ON p.pip_codigo = C.orgao_id
            LEFT JOIN NR_ARQUIVAMENTOS D ON p.pip_codigo = D.orgao_id
            LEFT JOIN NR_VISTAS_ABERTAS E ON p.pip_codigo = E.orgao_id
            LEFT JOIN NR_ACORDOS_N_PERSECUCAO F ON p.pip_codigo = F.orgao_id) t
    """.format(schema_exadata, schema_exadata_aux)
    ).withColumn(
        "dt_inclusao",
        from_unixtime(
            unix_timestamp(current_timestamp(), 'yyyy-MM-dd HH:mm:ss'), 'yyyy-MM-dd HH:mm:ss')
        .cast('timestamp')
    ).withColumn(
        "tamanho_periodo_dias",
        lit(tamanho_periodo_dias)
    )

    table_name = "{0}.{1}".format(schema_exadata_aux, output_table_name)

    table.write.mode("overwrite").saveAsTable("temp_table_{0}".format(output_table_name))
    temp_table = spark.table("temp_table_{0}".format(output_table_name))

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_{0}".format(output_table_name))

    execute_compute_stats(table_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table pip detalhe aproveitamentos")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    parser.add_argument('-p','--nbPastDays', metavar='nbPastDays', type=int, default=30, help='')
    parser.add_argument('-t','--tableName', metavar='tableName', type=str, default='tb_pip_detalhe_aproveitamentos', help='')
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort,
                    'nb_past_days': args.nbPastDays,
                    'table_name': args.tableName,
                }

    execute_process(options)
