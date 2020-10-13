# -*- coding: utf-8 -*- 
import argparse
import pyspark

from pyspark.sql import Window
from pyspark.sql.functions import max, col, count, concat_ws, collect_list, lit

from generic_utils import execute_compute_stats

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
   # ANDAMENTOS DE INTERESSE DA PIP
    DENUNCIA = (6252, 6253, 1201, 1202, 6254)
    ACORDO = (7914, 7928, 7883, 7827)
    DESACORDO = 7920
    ARQUIVAMENTO = (
        6549, 6593, 6591, 6343, 6338, 6339, 6340, 6341, 6342, 7871, 7897, 7912,
        6346, 6350, 6359, 6392, 6017, 6018, 6020, 7745
    )
    DESARQUIVAMENTO = (
        6075, 1028, 6798, 7245, 6307, 1027, 7803, 6003, 7802, 7801
    )
    CAUTELAR = (
        6648, 6649, 6650, 6651, 6652, 6653, 6654, 6038, 6039, 6040, 6041, 6042,
        6043, 7815, 7816, 6620, 6257, 6258, 7878, 7877, 6367, 6368, 6369, 6370,
        1208,1030
    )
    ANDAMENTOS_IMPORTANTES = (
        DENUNCIA + ACORDO + ARQUIVAMENTO + CAUTELAR
    )
    CANCELAMENTOS = (DESACORDO,) + DESARQUIVAMENTO

    """
    Regras:
    Denuncias e Cautelares - Soma simples de tudo que aparece em dado periodo.
    Acordos e Arquivamentos - Apenas considera 1. Por exemplo, caso haja mais de um
      acordo em um documento, contar apenas 1 vez. Ademais, se o ultimo movimento de
      acordo for de rescisao (ou desarquivamento no caso de arquivamentos) a soma fica zerada.
    """

    spark.sql("SELECT pip_codigo, aisp_codigo, aisp_nome FROM {0}.tb_pip_aisp".format(schema_exadata_aux)).createOrReplaceTempView("TABELA_PIP_AISP")
    spark.catalog.cacheTable("TABELA_PIP_AISP")

    spark.sql(
        """
        SELECT
        docu_dk,
        docu_nr_mp,
        pip_codigo,
        vist_dk,
        vist_docu_dk,
        vist_dt_abertura_vista,
        vist_dt_fechamento_vista
        FROM {0}.mcpr_documento
        JOIN {0}.mcpr_vista ON vist_docu_dk = docu_dk
        JOIN (SELECT DISTINCT pip_codigo FROM TABELA_PIP_AISP) TPA ON TPA.pip_codigo = vist_orgi_orga_dk
        WHERE docu_cldc_dk IN (3, 494, 590) -- PIC e Inqueritos
	AND docu_tpst_dk != 11 -- Documento nao cancelado
    """.format(schema_exadata)).createOrReplaceTempView(
            "VISTAS_FILTRADAS_SEM_ANDAMENTO"
    )

    spark.sql(
        """
        SELECT
            FSA.*,
            ANDAMENTO.pcao_dt_andamento,
            SUBANDAMENTO.stao_tppr_dk,
        CASE
            WHEN stao_tppr_dk in {DENUNCIA} THEN 4 -- denuncia
            WHEN stao_tppr_dk in {CAUTELAR} THEN 3 -- cautelar
            WHEN stao_tppr_dk in {ACORDO} THEN 2 -- acordo
            WHEN stao_tppr_dk in {ARQUIVAMENTO} THEN 1 -- arquivamento
        END as peso_prioridade
            FROM VISTAS_FILTRADAS_SEM_ANDAMENTO FSA
        JOIN {0}.mcpr_andamento ANDAMENTO ON pcao_vist_dk = vist_dk
        JOIN {0}.mcpr_sub_andamento SUBANDAMENTO ON stao_pcao_dk = pcao_dk
        WHERE pcao_dt_cancelamento IS NULL -- Andamento nao cancelado
        AND pcao_dt_andamento > cast(date_sub(current_timestamp(), {1}) as timestamp)
        AND stao_tppr_dk IN {ANDAMENTOS_IMPORTANTES}
        """.format(
                schema_exadata,
                days_ago,
                DENUNCIA=DENUNCIA,
                ARQUIVAMENTO=ARQUIVAMENTO,
                CAUTELAR=CAUTELAR,
                ACORDO=ACORDO,
                ANDAMENTOS_IMPORTANTES=ANDAMENTOS_IMPORTANTES
            )
    ).createOrReplaceTempView("ANDAMENTOS_IMPORTANTES")

    spark.sql(
        """
        SELECT
            FSA.*,
            ANDAMENTO.pcao_dt_andamento,
            SUBANDAMENTO.stao_tppr_dk,
        CASE
            WHEN stao_tppr_dk = {DESACORDO} THEN -2 -- desacordo
            WHEN stao_tppr_dk in {DESARQUIVAMENTO} THEN -1 -- desarquivamento
        END as peso_prioridade --Quanto maior mais importante
            FROM VISTAS_FILTRADAS_SEM_ANDAMENTO FSA
        JOIN {0}.mcpr_andamento ANDAMENTO ON pcao_vist_dk = vist_dk
        JOIN {0}.mcpr_sub_andamento SUBANDAMENTO ON stao_pcao_dk = pcao_dk
        WHERE pcao_dt_cancelamento IS NULL -- Andamento nao cancelado
        AND pcao_dt_andamento > cast(date_sub(current_timestamp(), {1}) as timestamp)
        AND stao_tppr_dk IN {CANCELAMENTOS}
        """.format(
            schema_exadata,
            days_ago,
            DESACORDO=DESACORDO,
            DESARQUIVAMENTO=DESARQUIVAMENTO,
            CANCELAMENTOS=CANCELAMENTOS
        )
    ).createOrReplaceTempView("DOCUMENTO_CANCELAMENTOS")

    spark.sql(
        """
        SELECT D.pip_codigo as orgao_id,
            SUM(CASE WHEN D.peso_prioridade = 1 THEN 1 ELSE 0 END) AS nr_arquivamentos,
            SUM(CASE WHEN D.peso_prioridade = 2 THEN 1 ELSE 0 END) AS nr_acordos_n_persecucao,
            SUM(CASE WHEN D.peso_prioridade = 3 THEN 1 ELSE 0 END) AS nr_cautelares,
            SUM(CASE WHEN D.peso_prioridade = 4 THEN 1 ELSE 0 END) AS nr_denuncias
        FROM (SELECT DISTINCT pip_codigo, docu_dk, pcao_dt_andamento, peso_prioridade FROM ANDAMENTOS_IMPORTANTES) D
        LEFT JOIN DOCUMENTO_CANCELAMENTOS C ON C.pip_codigo = D.pip_codigo
            AND C.docu_dk = D.docu_dk
            AND C.pcao_dt_andamento >= D.pcao_dt_andamento
            AND C.peso_prioridade + D.peso_prioridade = 0
        WHERE C.peso_prioridade IS NULL
        GROUP BY D.pip_codigo
        """
    ).createOrReplaceTempView("CONTAGENS")

   # A baixa a DP vai ser o numero de vistas abertas subtraída do
   # total de ANDAMENTOS IMPORTANTES desambiguados(sem repetição de andameto por mesma vista)
    spark.sql("""
	SELECT VA.pip_codigo as orgao_id,
	COUNT(DISTINCT VA.vist_dk) as nr_baixa_dp
        FROM VISTAS_FILTRADAS_SEM_ANDAMENTO VA
        LEFT JOIN (SELECT * FROM ANDAMENTOS_IMPORTANTES WHERE stao_tppr_dk IN {FINALIZACOES}) FID ON VA.vist_dk = FID.vist_dk
        WHERE stao_tppr_dk IS NULL -- vista sem ANDAMENTOS_IMPORTANTES
        AND VA.vist_dt_abertura_vista >= cast(date_sub(current_timestamp(), {0}) as timestamp)
	GROUP BY VA.pip_codigo
    """.format(days_ago, FINALIZACOES=ANDAMENTOS_IMPORTANTES+CANCELAMENTOS)).createOrReplaceTempView("NR_BAIXA_DP")

    metricas = spark.sql(
        """
        SELECT orgao_id, nm_orgao, cod_pct,
	    CONCAT_WS(', ', collect_list(aisp_codigo)) as aisp_codigo,
	    CONCAT_WS(', ', collect_list(aisp_nome)) as aisp_nome,
            MAX(nr_denuncias) as nr_denuncias,
            MAX(nr_cautelares) as nr_cautelares,
            MAX(nr_acordos_n_persecucao) as nr_acordos_n_persecucao,
            MAX(nr_arquivamentos) as nr_arquivamentos,
            MAX(nr_baixa_dp) as nr_aberturas_vista,
	        MAX(MAX(nr_denuncias)) OVER (PARTITION BY cod_pct) AS max_denuncias,
	        MAX(MAX(nr_cautelares)) OVER(PARTITION BY cod_pct) as max_cautelares,
            MAX(MAX(nr_acordos_n_persecucao)) OVER(PARTITION BY cod_pct) as max_acordos,
            MAX(MAX(nr_arquivamentos)) OVER(PARTITION BY cod_pct) as max_arquivamentos,
            MAX(MAX(nr_baixa_dp)) OVER(PARTITION BY cod_pct) as max_vistas,
            PERCENTILE(MAX(nr_denuncias), 0.5) OVER(PARTITION BY cod_pct) as med_denuncias,
            PERCENTILE(MAX(nr_cautelares), 0.5) OVER(PARTITION BY cod_pct) as med_cautelares,
            PERCENTILE(MAX(nr_acordos_n_persecucao), 0.5) OVER(PARTITION BY cod_pct) as med_acordos,
            PERCENTILE(MAX(nr_arquivamentos), 0.5) OVER(PARTITION BY cod_pct) as med_arquivamentos,
            PERCENTILE(MAX(nr_baixa_dp), 0.5) OVER(PARTITION BY cod_pct) as med_vistas
        FROM (
            SELECT
                p.pip_codigo as orgao_id, O.orgi_nm_orgao as nm_orgao, PCT.cod_pct,
                p.aisp_codigo as aisp_codigo,
                p.aisp_nome as aisp_nome,
                nvl(nr_denuncias, 0) as nr_denuncias,
                nvl(nr_cautelares, 0) as nr_cautelares,
                nvl(nr_acordos_n_persecucao, 0) as nr_acordos_n_persecucao,
                nvl(nr_arquivamentos, 0) as nr_arquivamentos,
                nvl(nr_baixa_dp, 0) as nr_baixa_dp
            FROM (SELECT DISTINCT pip_codigo, aisp_codigo, aisp_nome FROM TABELA_PIP_AISP) p
            JOIN {0}.orgi_orgao O ON orgi_dk = p.pip_codigo
            LEFT JOIN {1}.atualizacao_pj_pacote PCT ON p.pip_codigo = PCT.id_orgao
            LEFT JOIN CONTAGENS A ON p.pip_codigo = A.orgao_id
            LEFT JOIN NR_BAIXA_DP E ON p.pip_codigo = E.orgao_id) t
	    GROUP BY orgao_id, nm_orgao, cod_pct
    """.format(schema_exadata, schema_exadata_aux))
    metricas.createOrReplaceTempView("metricas")
    spark.catalog.cacheTable("metricas")

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
                   current_timestamp() as dt_calculo,
                   nm_max_denuncias,
                   nm_max_cautelares,
                   nm_max_acordos,
                   nm_max_arquivamentos,
                   nm_max_aberturas,
                   mt.cod_pct
            FROM metricas mt
            JOIN (SELECT cod_pct, MAX(nm_orgao) AS nm_max_denuncias FROM metricas WHERE nr_denuncias = max_denuncias GROUP BY cod_pct) NMD ON NMD.cod_pct = mt.cod_pct
            JOIN (SELECT cod_pct, MAX(nm_orgao) AS nm_max_cautelares FROM metricas WHERE nr_cautelares = max_cautelares GROUP BY cod_pct) NMC ON NMC.cod_pct = mt.cod_pct
            JOIN (SELECT cod_pct, MAX(nm_orgao) AS nm_max_acordos FROM metricas WHERE nr_acordos_n_persecucao = max_acordos GROUP BY cod_pct) NMA ON NMA.cod_pct = mt.cod_pct
            JOIN (SELECT cod_pct, MAX(nm_orgao) AS nm_max_arquivamentos FROM metricas WHERE nr_arquivamentos = max_arquivamentos GROUP BY cod_pct) NMAR ON NMAR.cod_pct = mt.cod_pct
            JOIN (SELECT cod_pct, MAX(nm_orgao) AS nm_max_aberturas FROM metricas WHERE nr_aberturas_vista = max_vistas GROUP BY cod_pct) NMAV ON NMAV.cod_pct = mt.cod_pct
    """)

    table_name = "{0}.{1}".format(schema_exadata_aux, output_table_name)
    stats.write.mode("overwrite").saveAsTable(
        "temp_table_{0}".format(output_table_name)
    )
    temp_table = spark.table("temp_table_{0}".format(output_table_name))

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_{0}".format(output_table_name))

    spark.catalog.clearCache()

    execute_compute_stats(table_name)


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
	    "table_name": args.tableName,
    }

    execute_process(options)
