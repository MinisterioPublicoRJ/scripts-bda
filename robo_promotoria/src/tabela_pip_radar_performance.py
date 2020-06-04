import pyspark
from pyspark.sql import Window
from pyspark.sql.functions import max, col, count, concat_ws, collect_list, lit
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
    ANDAMENTOS_FILTERED.createOrReplaceTempView("ANDAMENTOS_FILTERED")
    spark.catalog.cacheTable("ANDAMENTOS_FILTERED")

    spark.sql("SELECT pip_codigo, aisp_codigo, aisp_nome FROM {0}.tb_pip_aisp".format(schema_exadata_aux)).createOrReplaceTempView("TABELA_PIP_AISP")
    spark.catalog.cacheTable("TABELA_PIP_AISP")

    spark.sql("""
	SELECT stao_tppr_dk, stao_pcao_dk
        FROM {0}.mcpr_sub_andamento
	WHERE stao_tppr_dk IN (6648,6649,6650,6651,6652,6653,6654,6038,6039,6040,6041,6042,6043,
	7815,7816,6620,6257,6258,7878,7877,6367,6368,6369,6370,1208,1030,
	7827,7914,7883,7868,6361,6362,6391,7922,7928,7915,7917,7920,6549,
	6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,6346,6350,
	6359,6392,6017,6018,6020,7745,6075,1028,6798,7245,6307,1027,7803,
	6003,7802,7801,6252,6253,1201,1202,6254)
	""".format(schema_exadata)).createOrReplaceTempView("SUB_ANDAMENTOS")

    spark.catalog.cacheTable("SUB_ANDAMENTOS")

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
            SELECT stao_tppr_dk, stao_pcao_dk
            FROM SUB_ANDAMENTOS
            WHERE stao_tppr_dk IN (7914,7928,7883,7827,7920) --7920 rescisao do acordo
            ) D
            ON D.stao_pcao_dk = C.pcao_dk
        INNER JOIN (SELECT DISTINCT pip_codigo FROM TABELA_PIP_AISP) p
            ON p.pip_codigo = B.vist_orgi_orga_dk
        WHERE docu_cldc_dk IN (3, 494, 590)
        AND docu_tpst_dk != 11
    """.format(
            schema_exadata, schema_exadata_aux
        )
    )
    DOC_ACORDOS.createOrReplaceTempView("DOC_ACORDOS")
    spark.catalog.cacheTable("DOC_ACORDOS")

    MAX_DT_ACORD = spark.sql(
        """
        SELECT
            docu_dk,
            MAX(pcao_dt_andamento) AS max_dt_acordo
        FROM DOC_ACORDOS
        GROUP BY docu_dk
    """
    )
    MAX_DT_ACORD.createOrReplaceTempView("MAX_DT_ACORD")

    ACORDOS_POR_DOC = spark.sql(
        """
        SELECT
            docu_dk,
            vist_orgi_orga_dk,
            MAX(acordo) as acordos
        FROM (
            SELECT
            CASE WHEN pcao_dt_andamento >= cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (7914,7928,7883,7827)
                THEN 1 ELSE 0 END as acordo,
            A.docu_dk, A.vist_orgi_orga_dk, A.pcao_dt_andamento, A.stao_tppr_dk
            FROM DOC_ACORDOS A
            JOIN MAX_DT_ACORD MDT ON MDT.docu_dk = A.docu_dk AND MDT.max_dt_acordo = A.pcao_dt_andamento) t
        GROUP BY docu_dk, vist_orgi_orga_dk
    """.format(
            days_ago
        )
    )
    ACORDOS_POR_DOC.createOrReplaceTempView("ACORDOS_POR_DOC")
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
    NR_ACORDOS.createOrReplaceTempView("NR_ACORDOS")

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
            SELECT stao_tppr_dk, stao_pcao_dk
            FROM SUB_ANDAMENTOS
            WHERE stao_tppr_dk IN (7914,7928,7883,7827,7920) --7920 rescisao do acordo
            ) D
            ON D.stao_pcao_dk = C.pcao_dk
        INNER JOIN (SELECT DISTINCT pip_codigo FROM TABELA_PIP_AISP) p
            ON p.pip_codigo = B.vist_orgi_orga_dk
        WHERE docu_cldc_dk IN (3, 494, 590)
        AND docu_tpst_dk != 11
    """.format(
            schema_exadata, schema_exadata_aux
        )
    )
    DOC_ACORDOS_N_PERSECUCAO.createOrReplaceTempView("DOC_ACORDOS_N_PERSECUCAO")
    MAX_DT_ACORD_N_PERSECUCAO = spark.sql(
        """
        SELECT
            docu_dk,
            MAX(pcao_dt_andamento) AS max_dt_acordo
        FROM DOC_ACORDOS_N_PERSECUCAO
        GROUP BY docu_dk
    """
    )
    MAX_DT_ACORD_N_PERSECUCAO.createOrReplaceTempView("MAX_DT_ACORD_N_PERSECUCAO")
    ACORDOS_POR_DOC_N_PERSECUCAO = spark.sql(
        """
        SELECT
            docu_dk,
            vist_orgi_orga_dk,
            MAX(acordo_n_persecucao) as acordos_n_persecucao
        FROM (
            SELECT
            CASE WHEN pcao_dt_andamento > cast(date_sub(current_timestamp(), {0}) as timestamp)
                      AND stao_tppr_dk IN (7914,7928,7883,7827)
                THEN 1 ELSE 0 END as acordo_n_persecucao,
            A.docu_dk, A.vist_orgi_orga_dk, A.pcao_dt_andamento, A.stao_tppr_dk
            FROM DOC_ACORDOS_N_PERSECUCAO A
            JOIN MAX_DT_ACORD_N_PERSECUCAO MDT ON MDT.docu_dk = A.docu_dk AND MDT.max_dt_acordo = A.pcao_dt_andamento) t
        GROUP BY docu_dk, vist_orgi_orga_dk
    """.format(
            days_ago
        )
    )
    ACORDOS_POR_DOC_N_PERSECUCAO.createOrReplaceTempView(
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
    NR_ACORDOS_N_PERSECUCAO.createOrReplaceTempView("NR_ACORDOS_N_PERSECUCAO")

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
            SELECT stao_tppr_dk, stao_pcao_dk
            FROM SUB_ANDAMENTOS
            WHERE stao_tppr_dk IN (
                6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,6346,6350,6359,6392,6017,6018,6020,7745, --arquivamentos
                6075,1028,6798,7245,6307,1027,7803,6003,7802,7801) --desarquivamentos
            ) D
            ON D.stao_pcao_dk = C.pcao_dk
        INNER JOIN (SELECT DISTINCT pip_codigo FROM TABELA_PIP_AISP) p
            ON p.pip_codigo = B.vist_orgi_orga_dk
        WHERE docu_cldc_dk IN (3, 494, 590)
        AND docu_tpst_dk != 11
    """.format(
            schema_exadata, schema_exadata_aux
        )
    )
    DOC_ARQUIVAMENTOS.createOrReplaceTempView("DOC_ARQUIVAMENTOS")
    MAX_DT_ARQUIV = spark.sql(
        """
        SELECT
            docu_dk,
            MAX(pcao_dt_andamento) AS max_dt_arquivamento
        FROM DOC_ARQUIVAMENTOS
        GROUP BY docu_dk
    """
    )
    MAX_DT_ARQUIV.createOrReplaceTempView("MAX_DT_ARQUIV")
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
    ARQUIVAMENTOS_POR_DOC.createOrReplaceTempView("ARQUIVAMENTOS_POR_DOC")
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
    NR_ARQUIVAMENTOS.createOrReplaceTempView("NR_ARQUIVAMENTOS")

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
        INNER JOIN (SELECT DISTINCT pip_codigo FROM TABELA_PIP_AISP) p ON p.pip_codigo = t.orgao_id
        GROUP BY orgao_id
    """.format(
            schema_exadata, schema_exadata_aux, days_ago
        )
    )
    NR_CAUTELARES.createOrReplaceTempView("NR_CAUTELARES")

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
                SELECT stao_tppr_dk, stao_pcao_dk
                FROM SUB_ANDAMENTOS
                WHERE stao_tppr_dk IN (6252,6253,1201,1202,6254) --Denuncias
                    ) D
            ON D.stao_pcao_dk = C.pcao_dk
            WHERE docu_cldc_dk IN (3, 494, 590)
            AND docu_tpst_dk != 11
        ) t
        INNER JOIN (SELECT DISTINCT pip_codigo FROM TABELA_PIP_AISP) p ON p.pip_codigo = t.orgao_id
        GROUP BY orgao_id
    """.format(
            schema_exadata, schema_exadata_aux, days_ago
        )
    )
    NR_DENUNCIAS.createOrReplaceTempView("NR_DENUNCIAS")

    # Numero de vistas abertas
    NR_VISTAS_ABERTAS = spark.sql(
        """
        SELECT
            vist_orgi_orga_dk as orgao_id,
            COUNT(vist_dk) as nr_aberturas_vista
        FROM {0}.mcpr_documento
        JOIN {0}.mcpr_vista ON vist_docu_dk = docu_dk
        INNER JOIN (SELECT DISTINCT pip_codigo FROM TABELA_PIP_AISP) p 
            ON p.pip_codigo = vist_orgi_orga_dk
        WHERE docu_cldc_dk IN (3, 494, 590) -- PIC e Inqueritos
        AND vist_dt_abertura_vista >= cast(date_sub(current_timestamp(), {2}) as timestamp)
        AND docu_tpst_dk != 11
        GROUP BY vist_orgi_orga_dk
    """.format(
            schema_exadata, schema_exadata_aux, days_ago
        )
    )
    NR_VISTAS_ABERTAS.createOrReplaceTempView("NR_VISTAS_ABERTAS")
    metricas = spark.sql(
        """
        SELECT orgao_id, nm_orgao,
	    CONCAT_WS(', ', collect_list(aisp_codigo)) as aisp_codigo,
	    CONCAT_WS(', ', collect_list(aisp_nome)) as aisp_nome,
            MAX(nr_denuncias) as nr_denuncias,
            MAX(nr_cautelares) as nr_cautelares,
            MAX(nr_acordos_n_persecucao) as nr_acordos_n_persecucao,
            MAX(nr_arquivamentos) as nr_arquivamentos,
            MAX(nr_aberturas_vista) as nr_aberturas_vista,
	    MAX(MAX(nr_denuncias)) OVER () AS max_denuncias,
	    MAX(MAX(nr_cautelares)) OVER() as max_cautelares,
            MAX(MAX(nr_acordos_n_persecucao)) OVER() as max_acordos,
            MAX(MAX(nr_arquivamentos)) OVER() as max_arquivamentos,
            MAX(MAX(nr_aberturas_vista)) OVER() as max_vistas,
            PERCENTILE(MAX(nr_denuncias), 0.5) OVER() as med_denuncias,
            PERCENTILE(MAX(nr_cautelares), 0.5) OVER() as med_cautelares,
            PERCENTILE(MAX(nr_acordos_n_persecucao), 0.5) OVER() as med_acordos,
            PERCENTILE(MAX(nr_arquivamentos), 0.5) OVER() as med_arquivamentos,
            PERCENTILE(MAX(nr_aberturas_vista), 0.5) OVER() as med_vistas
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
            FROM (SELECT DISTINCT pip_codigo, aisp_codigo, aisp_nome FROM TABELA_PIP_AISP) p
            JOIN {0}.orgi_orgao O ON orgi_dk = p.pip_codigo
            LEFT JOIN NR_DENUNCIAS A ON p.pip_codigo = A.orgao_id
            LEFT JOIN NR_CAUTELARES B ON p.pip_codigo= B.orgao_id
            LEFT JOIN NR_ACORDOS C ON p.pip_codigo = C.orgao_id
            LEFT JOIN NR_ARQUIVAMENTOS D ON p.pip_codigo = D.orgao_id
            LEFT JOIN NR_VISTAS_ABERTAS E ON p.pip_codigo = E.orgao_id
            LEFT JOIN NR_ACORDOS_N_PERSECUCAO F ON p.pip_codigo = F.orgao_id) t
	    GROUP BY orgao_id, nm_orgao
    """.format(schema_exadata, schema_exadata_aux))
    metricas.createOrReplaceTempView("metricas")

    spark.catalog.cacheTable("metricas")

    orgao_max_den = (
        metricas.where(col("nr_denuncias") == col("max_denuncias"))
        .select(["orgao_id", "nm_orgao"])
	.groupBy("orgao_id")
	.agg(
	    concat_ws(", ", collect_list("nm_orgao")).alias("nm_max_denuncias")
	)
    ).collect()[0][1]
    orgao_max_cau = (
        metricas.where(col("nr_cautelares") == col("max_cautelares"))
        .select(["orgao_id", "nm_orgao"])
	.groupBy("orgao_id")
	.agg(
	    concat_ws(", ", collect_list("nm_orgao")).alias("nm_max_cautelares")
	)
    ).collect()[0][1]
    orgao_max_aco = (
        metricas.where(col("nr_acordos_n_persecucao") == col("max_acordos"))
        .select(["orgao_id", "nm_orgao"])
	.groupBy("orgao_id")
	.agg(
	    concat_ws(", ", collect_list("nm_orgao")).alias("nm_max_acordos")
	)
    ).collect()[0][1]
    orgao_max_arq = (
        metricas.where(col("nr_arquivamentos") == col("max_arquivamentos"))
        .select(["orgao_id", "nm_orgao"])
	.groupBy("orgao_id")
	.agg(
	    concat_ws(", ", collect_list("nm_orgao")).alias("nm_max_arquivamentos")
	)
    ).collect()[0][1]
    orgao_max_abe = (
        metricas.where(col("nr_aberturas_vista") == col("max_vistas"))
        .select(["orgao_id", "nm_orgao"])
	.groupBy("orgao_id")
	.agg(
	    concat_ws(", ", collect_list("nm_orgao")).alias("nm_max_vistas")
	)
    ).collect()[0][1]
    stats = spark.sql(
            """
            SELECT mt.aisp_codigo,
                   mt.aisp_nome,
                   mt.orgao_id,
                   nr_denuncias,
                   nr_cautelares,
                   nr_acordos_n_persecucao,
                   nr_arquivamentos,
                   nr_aberturas_vista,
                   max_denuncias,
                   max_cautelares,
                   max_acordos,
                   max_arquivamentos,
                   max_vistas,
                   nr_denuncias / max_denuncias
                       as perc_denuncias,
                   nr_cautelares / max_cautelares
                       as perc_cautelares,
                   nr_acordos_n_persecucao / max_acordos
                       as perc_acordos,
                   nr_arquivamentos / max_arquivamentos as perc_arquivamentos,
                   nr_aberturas_vista / max_vistas as perc_aberturas_vista,
                   med_denuncias,
                   med_cautelares,
                   med_acordos,
                   med_arquivamentos,
                   med_vistas,
                   (nr_denuncias - med_denuncias)
                       / med_denuncias as var_med_denuncias,
                   (nr_cautelares - med_cautelares)
                       / med_cautelares as var_med_cautelares,
                   (nr_acordos_n_persecucao - med_acordos)
                       / med_acordos as var_med_acordos,
                   (nr_arquivamentos - med_arquivamentos)
                      / med_arquivamentos as var_med_arquivamentos,
                   (nr_aberturas_vista - med_vistas)
                       / med_vistas as var_med_aberturas_vista,
                   current_timestamp() as dt_calculo
            FROM metricas mt
    """)\
	.withColumn("nm_max_denuncias", lit(orgao_max_den))\
	.withColumn("nm_max_cautelares", lit(orgao_max_cau))\
	.withColumn("nm_max_acordos", lit(orgao_max_aco))\
	.withColumn("nm_max_arquivamentos", lit(orgao_max_arq))\
	.withColumn("nm_max_aberturas", lit(orgao_max_abe))

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
    spark.catalog.clearCache()


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
	"table_name": args.tableName
    }

    execute_process(options)
