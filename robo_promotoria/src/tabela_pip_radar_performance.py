import pyspark
from pyspark.sql import Window
from pyspark.sql.functions import max, col, count, concat_ws, collect_list
from utils import _update_impala_table
import argparse


def execute_process(options):

    spark = (
        pyspark.sql.session.SparkSession.builder.appName(
            "criar_tabela_pip_radar_performance"
        )
        .enableHiveSupport()
        .getOrCreate()
    )
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

    schema_exadata = options["schema_exadata"]
    schema_exadata_aux = options["schema_exadata_aux"]

    days_ago = options["days_ago"]

    output_table_name = options["table_name"]

    """
    Regras:
    Denuncias e Cautelares - Soma simples de tudo que aparece em dado periodo.
    Acordos e Arquivamentos - Apenas considera 1. Por exemplo, caso haja mais de um
      acordo em um documento, contar apenas 1 vez. Ademais, se o ultimo movimento de
      acordo for de rescisao (ou desarquivamento no caso de arquivamentos) a soma fica zerada.
    """

    ANDAMENTOS_FILTERED = spark.sql(
        """
        SELECT *
        FROM {0}.mcpr_andamento s
        WHERE s.year_month >= cast(date_format(date_sub(current_timestamp(), {1}), 'yyyyMM') as INT)
        and pcao_dt_andamento > cast(date_sub(current_timestamp(), {1}) as timestamp)
        and pcao_dt_andamento <= current_timestamp()
    """.format(
            schema_exadata, days_ago
        )
    )
    ANDAMENTOS_FILTERED.registerTempTable("ANDAMENTOS_FILTERED")

    # Numero de acordos
    DOC_ACORDOS = spark.sql(
        """
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
    """.format(
            schema_exadata, schema_exadata_aux
        )
    )
    DOC_ACORDOS.registerTempTable("DOC_ACORDOS")
    MAX_DT_ACORD = spark.sql(
        """
        SELECT
            docu_dk,
            MAX(pcao_dt_andamento) AS max_dt_acordo
        FROM DOC_ACORDOS
        GROUP BY docu_dk
    """
    )
    MAX_DT_ACORD.registerTempTable("MAX_DT_ACORD")
    ACORDOS_POR_DOC = spark.sql(
        """
        SELECT
            docu_dk,
            vist_orgi_orga_dk,
            MAX(acordo) as acordos
        FROM (
            SELECT
            CASE WHEN pcao_dt_andamento >= cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (7827,7914,7883,7868,6361,6362,6391,7922,7928,7915,7917)
                THEN 1 ELSE 0 END as acordo,
            A.docu_dk, A.vist_orgi_orga_dk, A.pcao_dt_andamento, A.stao_tppr_dk
            FROM DOC_ACORDOS A
            JOIN MAX_DT_ACORD MDT ON MDT.docu_dk = A.docu_dk AND MDT.max_dt_acordo = A.pcao_dt_andamento) t
        GROUP BY docu_dk, vist_orgi_orga_dk
    """.format(
            days_ago
        )
    )
    ACORDOS_POR_DOC.registerTempTable("ACORDOS_POR_DOC")
    NR_ACORDOS = spark.sql(
        """
        SELECT
            vist_orgi_orga_dk as orgao_id,
            SUM(acordos) as nr_acordos
        FROM ACORDOS_POR_DOC
        GROUP BY vist_orgi_orga_dk
    """.format(
            schema_exadata
        )
    )
    NR_ACORDOS.registerTempTable("NR_ACORDOS")

    # Numero de acordos de nao persecucao
    DOC_ACORDOS_N_PERSECUCAO = spark.sql(
        """
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
    """.format(
            schema_exadata, schema_exadata_aux
        )
    )
    DOC_ACORDOS_N_PERSECUCAO.registerTempTable("DOC_ACORDOS_N_PERSECUCAO")
    MAX_DT_ACORD_N_PERSECUCAO = spark.sql(
        """
        SELECT
            docu_dk,
            MAX(pcao_dt_andamento) AS max_dt_acordo
        FROM DOC_ACORDOS_N_PERSECUCAO
        GROUP BY docu_dk
    """
    )
    MAX_DT_ACORD_N_PERSECUCAO.registerTempTable("MAX_DT_ACORD_N_PERSECUCAO")
    ACORDOS_POR_DOC_N_PERSECUCAO = spark.sql(
        """
        SELECT
            docu_dk,
            vist_orgi_orga_dk,
            MAX(acordo_n_persecucao) as acordos_n_persecucao
        FROM (
            SELECT
            CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (7914,7917,7928,7883,7915,7922,7827)
                THEN 1 ELSE 0 END as acordo_n_persecucao,
            A.docu_dk, A.vist_orgi_orga_dk, A.pcao_dt_andamento, A.stao_tppr_dk
            FROM DOC_ACORDOS_N_PERSECUCAO A
            JOIN MAX_DT_ACORD_N_PERSECUCAO MDT ON MDT.docu_dk = A.docu_dk AND MDT.max_dt_acordo = A.pcao_dt_andamento) t
        GROUP BY docu_dk, vist_orgi_orga_dk
    """.format(
            days_ago
        )
    )
    ACORDOS_POR_DOC_N_PERSECUCAO.registerTempTable(
        "ACORDOS_POR_DOC_N_PERSECUCAO"
    )
    NR_ACORDOS_N_PERSECUCAO = spark.sql(
        """
        SELECT
            vist_orgi_orga_dk as orgao_id,
            SUM(acordos_n_persecucao) as nr_acordos_n_persecucao
        FROM ACORDOS_POR_DOC_N_PERSECUCAO
        GROUP BY vist_orgi_orga_dk
    """.format(
            schema_exadata
        )
    )
    NR_ACORDOS_N_PERSECUCAO.registerTempTable("NR_ACORDOS_N_PERSECUCAO")

    # Numero de Arquivamentos
    DOC_ARQUIVAMENTOS = spark.sql(
        """
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
    """.format(
            schema_exadata, schema_exadata_aux
        )
    )
    DOC_ARQUIVAMENTOS.registerTempTable("DOC_ARQUIVAMENTOS")
    MAX_DT_ARQUIV = spark.sql(
        """
        SELECT
            docu_dk,
            MAX(pcao_dt_andamento) AS max_dt_arquivamento
        FROM DOC_ARQUIVAMENTOS
        GROUP BY docu_dk
    """
    )
    MAX_DT_ARQUIV.registerTempTable("MAX_DT_ARQUIV")
    ARQUIVAMENTOS_POR_DOC = spark.sql(
        """
        SELECT
            docu_dk,
            vist_orgi_orga_dk,
            MAX(arquivamento) as arquivamentos
        FROM (
            SELECT
            CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,6346,6350,6359,6392,6017,6018,6020,7745)
                THEN 1 ELSE 0 END as arquivamento,
            A.docu_dk, A.vist_orgi_orga_dk, A.pcao_dt_andamento, A.stao_tppr_dk
            FROM DOC_ARQUIVAMENTOS A
            JOIN MAX_DT_ARQUIV MDT ON MDT.docu_dk = A.docu_dk AND MDT.max_dt_arquivamento = A.pcao_dt_andamento) t
        GROUP BY docu_dk, vist_orgi_orga_dk
    """.format(
            days_ago
        )
    )
    ARQUIVAMENTOS_POR_DOC.registerTempTable("ARQUIVAMENTOS_POR_DOC")
    NR_ARQUIVAMENTOS = spark.sql(
        """
        SELECT
            vist_orgi_orga_dk as orgao_id,
            SUM(arquivamentos) as nr_arquivamentos
        FROM ARQUIVAMENTOS_POR_DOC
        GROUP BY vist_orgi_orga_dk
    """.format(
            schema_exadata
        )
    )
    NR_ARQUIVAMENTOS.registerTempTable("NR_ARQUIVAMENTOS")

    # Numero de medidas cautelares
    NR_CAUTELARES = spark.sql(
        """
        SELECT
            orgao_id,
            SUM(cautelar) as nr_cautelares
        FROM (
            SELECT
                CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), {2}) as timestamp) THEN 1 ELSE 0 END as cautelar,
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
    """.format(
            schema_exadata, schema_exadata_aux, days_ago
        )
    )
    NR_CAUTELARES.registerTempTable("NR_CAUTELARES")

    # Numero de denuncias
    NR_DENUNCIAS = spark.sql(
        """
        SELECT 
            orgao_id,
            SUM(denuncia) as nr_denuncias
        FROM (
            SELECT 
                CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), {2}) as timestamp) THEN 1 ELSE 0 END as denuncia,
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
    """.format(
            schema_exadata, schema_exadata_aux, days_ago
        )
    )
    NR_DENUNCIAS.registerTempTable("NR_DENUNCIAS")

    # Numero de vistas abertas
    NR_VISTAS_ABERTAS = spark.sql(
        """
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
    """.format(
            schema_exadata, schema_exadata_aux, days_ago
        )
    )
    NR_VISTAS_ABERTAS.registerTempTable("NR_VISTAS_ABERTAS")
    spark.sql(
        """
        SELECT orgao_id, nm_orgao,
            aisp_codigo,
            aisp_nome,
            nr_denuncias,
            nr_cautelares,
            nr_acordos_n_persecucao,
            nr_arquivamentos,
            nr_aberturas_vista
        FROM (
            SELECT
                p.pip_codigo as orgao_id, O.orgi_nm_orgao as nm_orgao,
                p.aisp_codigo as aisp_codigo,
                p.aisp_nome as aisp_nome,
                nvl(nr_denuncias, 0) as nr_denuncias,
                nvl(nr_cautelares, 0) as nr_cautelares,
                nvl(nr_acordos_n_persecucao, 0) as nr_acordos_n_persecucao,
                nvl(nr_arquivamentos, 0) as nr_arquivamentos,
                nvl(nr_aberturas_vista, 0) as nr_aberturas_vista
            FROM (SELECT DISTINCT pip_codigo, aisp_codigo, aisp_nome FROM {1}.tb_pip_aisp) p
            JOIN {0}.orgi_orgao O ON orgi_dk = p.pip_codigo
            LEFT JOIN NR_DENUNCIAS A ON p.pip_codigo = A.orgao_id
            LEFT JOIN NR_CAUTELARES B ON p.pip_codigo= B.orgao_id
            LEFT JOIN NR_ACORDOS C ON p.pip_codigo = C.orgao_id
            LEFT JOIN NR_ARQUIVAMENTOS D ON p.pip_codigo = D.orgao_id
            LEFT JOIN NR_VISTAS_ABERTAS E ON p.pip_codigo = E.orgao_id
            LEFT JOIN NR_ACORDOS_N_PERSECUCAO F ON p.pip_codigo = F.orgao_id) t
    """.format(schema_exadata, schema_exadata_aux)).\
        registerTempTable("metricas")

    max_aisp = spark.sql("""
        SELECT aisp_codigo, nm_orgao,
            max(nr_denuncias) as max_den,
            max(nr_cautelares) as max_cau,
            max(nr_acordos_n_persecucao) as max_aco,
            max(nr_arquivamentos) as max_arq,
            max(nr_aberturas_vista) as max_abe
            from metricas
            GROUP BY aisp_codigo, nm_orgao
    """)
    w = Window.partitionBy("aisp_codigo")
    orgao_max_den = (
        max_aisp.withColumn("m_max_den", max("max_den").over(w))
        .where(col("max_den") == col("m_max_den"))
        .select(["aisp_codigo", "nm_orgao"])
        .groupBy("aisp_codigo")
        .agg(
            concat_ws(", ", collect_list("nm_orgao"))
            .alias("nm_max_denuncias")
        )
        .withColumnRenamed("aisp_codigo", "den_aisp_codigo")
    )
    orgao_max_cau = (
        max_aisp.withColumn("m_max_cau", max("max_cau").over(w))
        .where(col("max_cau") == col("m_max_cau"))
        .select(["aisp_codigo", "nm_orgao"])
        .groupBy("aisp_codigo")
        .agg(
            concat_ws(", ", collect_list("nm_orgao"))
            .alias("nm_max_cautelares")
        )
        .withColumnRenamed("aisp_codigo", "cau_aisp_codigo")
    )
    orgao_max_aco = (
        max_aisp.withColumn("m_max_aco", max("max_aco").over(w))
        .where(col("max_aco") == col("m_max_aco"))
        .select(["aisp_codigo", "nm_orgao"])
        .groupBy("aisp_codigo")
        .agg(
            concat_ws(", ", collect_list("nm_orgao"))
            .alias("nm_max_acordos")
        )
        .withColumnRenamed("aisp_codigo", "aco_aisp_codigo")
    )
    orgao_max_arq = (
        max_aisp.withColumn("m_max_arq", max("max_arq").over(w))
        .where(col("max_arq") == col("m_max_arq"))
        .select(["aisp_codigo", "nm_orgao"])
        .groupBy("aisp_codigo")
        .agg(
            concat_ws(", ", collect_list("nm_orgao"))
            .alias("nm_max_arquivamentos")
        )
        .withColumnRenamed("aisp_codigo", "arq_aisp_codigo")
    )
    orgao_max_abe = (
        max_aisp.withColumn("m_max_abe", max("max_abe").over(w))
        .where(col("max_abe") == col("m_max_abe"))
        .select(["aisp_codigo", "nm_orgao"])
        .groupBy("aisp_codigo")
        .agg(
            concat_ws(", ", collect_list("nm_orgao"))
            .alias("nm_max_aberturas")
        )
        .withColumnRenamed("aisp_codigo", "abe_aisp_codigo")
    )
    spark.sql(
        """
            SELECT aisp_codigo, max(nr_denuncias) as max_aisp_denuncias,
                   max(nr_cautelares) as max_aisp_cautelares,
                   max(nr_acordos_n_persecucao) as max_aisp_acordos,
                   max(nr_arquivamentos) as max_aisp_arquivamentos,
                   max(nr_aberturas_vista) as max_aisp_aberturas_vista,
                   percentile(nr_denuncias, 0.5) as med_aisp_denuncias,
                   percentile(nr_cautelares, 0.5) as med_aisp_cautelares,
                   percentile(nr_acordos_n_persecucao, 0.5) as med_aisp_acordos,
                   percentile(nr_arquivamentos,  0.5) as med_aisp_arquivamentos,
                   percentile(nr_aberturas_vista, 0.5) as med_aisp_aberturas
                   FROM metricas
                   GROUP BY aisp_codigo
    """
    ).createOrReplaceTempView("stats_aisp")
    stats = (
        spark.sql(
            """
            SELECT mt.aisp_codigo,
                   mt.aisp_nome,
                   mt.orgao_id,
                   nr_denuncias,
                   nr_cautelares,
                   nr_acordos_n_persecucao,
                   nr_arquivamentos,
                   nr_aberturas_vista
                   max_aisp_denuncias,
                   max_aisp_cautelares,
                   max_aisp_acordos,
                   max_aisp_arquivamentos,
                   max_aisp_aberturas_vista,
                   nr_denuncias / max_aisp_denuncias
                       as perc_denuncias,
                   nr_cautelares / max_aisp_cautelares
                       as perc_cautelares,
                   nr_acordos_n_persecucao / max_aisp_acordos
                       as perc_acordos,
                   nr_arquivamentos / max_aisp_arquivamentos as perc_arquivamentos,
                   nr_aberturas_vista / max_aisp_aberturas_vista as perc_aberturas,
                   med_aisp_denuncias,
                   med_aisp_cautelares,
                   med_aisp_acordos,
                   med_aisp_arquivamentos,
                   med_aisp_aberturas,
                   (nr_denuncias - med_aisp_denuncias)
                       / med_aisp_denuncias as var_med_denuncias,
                   (nr_cautelares - med_aisp_cautelares)
                       / med_aisp_cautelares as var_med_cautelares,
                   (nr_acordos_n_persecucao - med_aisp_acordos)
                       / med_aisp_acordos as var_med_acordos,
                   (nr_arquivamentos - med_aisp_arquivamentos)
                      / med_aisp_arquivamentos as var_med_arquivamentos,
                   (nr_aberturas_vista - med_aisp_aberturas)
                       / med_aisp_aberturas as var_med_aberturas,
                   current_timestamp() as dt_calculo
            FROM metricas mt
            INNER JOIN stats_aisp sa
            ON mt.aisp_codigo = sa.aisp_codigo
    """
        )
        .join(orgao_max_den, col("aisp_codigo") == col("den_aisp_codigo"))
        .drop("den_aisp_codigo")
        .join(orgao_max_cau, col("aisp_codigo") == col("cau_aisp_codigo"))
        .drop("cau_aisp_codigo")
        .join(orgao_max_aco, col("aisp_codigo") == col("aco_aisp_codigo"))
        .drop("aco_aisp_codigo")
        .join(orgao_max_arq, col("aisp_codigo") == col("arq_aisp_codigo"))
        .drop("arq_aisp_codigo")
        .join(orgao_max_abe, col("aisp_codigo") == col("abe_aisp_codigo"))
        .drop("abe_aisp_codigo")
    )

    table_name = "{0}.{1}".format(schema_exadata_aux, output_table_name)
    stats.write.mode("overwrite").saveAsTable(
        "temp_table_{0}".format(output_table_name)
    )
    temp_table = spark.table("temp_table_{0}".format(output_table_name))

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_{0}".format(output_table_name))

    _update_impala_table(
        table_name, options["impala_host"], options["impala_port"]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create table pip radar performance"
    )
    parser.add_argument(
        "-e", "--schemaExadata", metavar="schemaExadata", type=str, help=""
    )
    parser.add_argument(
        "-a",
        "--schemaExadataAux",
        metavar="schemaExadataAux",
        type=str,
        help="",
    )
    parser.add_argument(
        "-i", "--impalaHost", metavar="impalaHost", type=str, help=""
    )
    parser.add_argument(
        "-o", "--impalaPort", metavar="impalaPort", type=str, help=""
    )
    parser.add_argument(
        "-p",
        "--daysAgo",
        metavar="days_ago",
        type=int,
        default=180,
        help="",
    )
    parser.add_argument(
        "-t",
        "--tableName",
        metavar="tableName",
        type=str,
        default="tb_pip_radar_performance",
        help="",
    )
    args = parser.parse_args()

    options = {
        "schema_exadata": args.schemaExadata,
        "schema_exadata_aux": args.schemaExadataAux,
        "impala_host": args.impalaHost,
        "impala_port": args.impalaPort,
        "days_ago": args.daysAgo,
    }

    execute_process(options)
