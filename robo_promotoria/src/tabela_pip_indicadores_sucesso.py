#-*-coding:utf-8-*-
import argparse
from datetime import datetime

import pyspark


def execute_process(options):

    spark = (
        pyspark.sql.session.SparkSession.builder.appName(
            "criar_tabela_indicadores_sucesso"
        )
        .enableHiveSupport()
        .getOrCreate()
    )

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
        DENUNCIA + ACORDO + (DESACORDO,) + ARQUIVAMENTO + DESARQUIVAMENTO + CAUTELAR
    )

    schema_exadata = options["schema_exadata"]
    schema_exadata_aux = options["schema_exadata_aux"]

    # days_past_start = options["days_past_start"]
    # Estabelece dia 01/03/2020 como ponto zero de início.
    # Caso intervalo de 545 dias (ano e meio) passe dessa data, ele força o início nela.
    nb = (datetime.now() - datetime(2020, 3, 1)).days
    days_past_start = min(nb, 545)
    # days_past_end = options["days_past_end"]
    # Estabelece dia 01/03/2021 como ponto zero de fim.
    # Antes de chegar a esse dia, usa-se a data atual para o fim do intervalo.
    # Ao passar de 01/03/2021, o final do intervalo será nela.
    # Depois de 180 dias, o final do intervalo passa a se mover novamente.
    nb = (datetime.now() - datetime(2021, 3, 1)).days
    days_past_end = min(max(nb, 0), 180)

    spark.sql(
        """
        SELECT
        docu_dk,
        docu_nr_mp,
        pip_codigo,
        vist_dk,
        vist_docu_dk,
        vist_dt_abertura_vista,
        vist_dt_fechamento_vista,
        cdtipfunc
        FROM {0}.mcpr_documento
        JOIN {0}.mcpr_vista ON vist_docu_dk = docu_dk
        JOIN (
            SELECT pip_codigo as codigo, pip_codigo
            from {1}.tb_pip_aisp
            GROUP BY pip_codigo
        ) p
        ON p.codigo = vist_orgi_orga_dk
        JOIN {0}.mcpr_pessoa_fisica pess ON pess.pesf_pess_dk = vist_pesf_pess_dk_resp_andam
        JOIN {0}.rh_funcionario f ON pess.pesf_cpf = f.cpf
        WHERE docu_cldc_dk IN (3, 494, 590) -- PIC e Inqueritos
        AND (vist_dt_fechamento_vista >= cast(date_sub(current_timestamp(), {2}) as timestamp)
        OR vist_dt_fechamento_vista IS NULL)
	AND docu_tpst_dk != 11 -- Documento nao cancelado
	""".format(
            schema_exadata, schema_exadata_aux, days_past_start
	)).createOrReplaceTempView(
            "FILTRADOS_SEM_ANDAMENTO"
	)

    # Estamos contabilizando múltiplas vistas abertas no mesmo dia para o mesmo documento.
    # O objetivo aqui é ter o número de vistas abertas (mesmo sem andamento associado)
    # por órgão. Não importando se houve mais de uma abertura de vista no mesmo
    # documento no mesmo dia.
    #
    # Grupo: todo inquérito policial e PIC que teve pelo menos uma
    # vista aberta para o promotor entre 18 e 6 meses atrás.
    spark.sql(
        """
        SELECT pip_codigo as orgao_id,
        COUNT(DISTINCT docu_dk) as vistas
        FROM FILTRADOS_SEM_ANDAMENTO
        WHERE vist_dt_fechamento_vista <= cast(date_sub(current_timestamp(), {0}) as timestamp)
        AND cdtipfunc IN ('1', '2') -- Filtra por vistas abertas por PROMOTORES
        GROUP BY pip_codigo
        """.format(days_past_end)).createOrReplaceTempView("GRUPO")

    spark.sql(
        """SELECT DISTINCT docu_dk
            FROM FILTRADOS_SEM_ANDAMENTO
            WHERE vist_dt_fechamento_vista <= cast(date_sub(current_timestamp(), {0}) as timestamp)
        """.format(days_past_end)
    ).createOrReplaceTempView("DOCUMENTOS_GRUPO")

    # Ordem de prioridade para desambiguação se ocorrerem multiplas vistas
    # no mesmo dia (para mesmo órgão e documento):
    #       denúncia > cautelar > acordo > arquivamento

    # Tabela com todos os Documentos + Vistas + Andamentos/Sub-Andamentos
    # imortantes (denúncia, arquivamento, desarquivamento, acordo, desacordo)
    # já desambiguado pela ordem de prioridade supracitada e, consequentemente,
    # sem repetição de vistas abertas no mesmo Órgão, mesmo Documento e mesmo Dia.
    # É possível que existam Andamentos repetidos (com mesmo stao_tppr_dk)
    # no mesmo mesmo órgão, mesmo documento e mesmo dia, mas esse caso
    # é resolvido com  COUNT(DISTINCT docu_dk)

    spark.sql(
        """WITH ANDAMENTOS_IMPORTANTES AS (SELECT
            FSA.*,
            ANDAMENTO.pcao_dt_andamento,
            SUBANDAMENTO.stao_tppr_dk,
        CASE
            WHEN stao_tppr_dk in {DENUNCIA} THEN 'denunciado'
            WHEN stao_tppr_dk in {ACORDO} THEN 'acordado'
            WHEN stao_tppr_dk = {DESACORDO} THEN 'desacordado'
            WHEN stao_tppr_dk in {ARQUIVAMENTO} THEN 'arquivado'
            WHEN stao_tppr_dk in {DESARQUIVAMENTO} THEN 'desarquivado'
            WHEN stao_tppr_dk in {CAUTELAR} THEN 'cautelado'
        END as tipo,
        CASE
            WHEN stao_tppr_dk in {DENUNCIA} THEN 4 -- denuncia
            WHEN stao_tppr_dk in {CAUTELAR} THEN 3 -- cautelar
            WHEN stao_tppr_dk in {ACORDO} THEN 2.1 -- acordo
            WHEN stao_tppr_dk = {DESACORDO} THEN 2 -- desacordo
            WHEN stao_tppr_dk in {ARQUIVAMENTO} THEN 1.1 -- arquivamento
            WHEN stao_tppr_dk in {DESARQUIVAMENTO} THEN 1 -- desarquivamento
        END as peso_prioridade --Quanto maior mais importante
            FROM FILTRADOS_SEM_ANDAMENTO FSA
        JOIN {0}.mcpr_andamento ANDAMENTO ON pcao_vist_dk = vist_dk
        JOIN {0}.mcpr_sub_andamento SUBANDAMENTO ON stao_pcao_dk = pcao_dk
        JOIN DOCUMENTOS_GRUPO DG ON FSA.docu_dk = DG.docu_dk --  Essse join serve para filtrar os Documentos que estão dentro da tabela GRUPO
	WHERE pcao_dt_cancelamento IS NULL -- Andamento nao cancelado
        AND stao_tppr_dk IN {ANDAMENTOS_IMPORTANTES}) --cautelares part 2/2
        SELECT TA.* FROM ANDAMENTOS_IMPORTANTES TA
        JOIN (
            SELECT pip_codigo, docu_dk, MAX(pcao_dt_andamento) AS ultimo_andamento,
            MAX(peso_prioridade) as maxima_prioridade
            FROM ANDAMENTOS_IMPORTANTES GROUP BY pip_codigo, docu_dk) SUB_TA
        ON TA.pip_codigo = SUB_TA.pip_codigo AND TA.docu_dk = SUB_TA.docu_dk
        AND TA.pcao_dt_andamento = SUB_TA.ultimo_andamento
        AND TA.peso_prioridade = SUB_TA.maxima_prioridade
        """.format(
            schema_exadata,
            DENUNCIA=DENUNCIA,
            ARQUIVAMENTO=ARQUIVAMENTO,
            DESARQUIVAMENTO=DESARQUIVAMENTO,
            CAUTELAR=CAUTELAR,
            ACORDO=ACORDO,
            DESACORDO=DESACORDO,
            ANDAMENTOS_IMPORTANTES=ANDAMENTOS_IMPORTANTES
        )
    ).createOrReplaceTempView("FILTRADOS_IMPORTANTES_DESAMBIGUADOS")

    spark.sql(
        """
        SELECT
            pip_codigo as orgao_id,
            COUNT(DISTINCT docu_dk) as denuncias --distinct docu_dk para evitar andamentos duplicados no mesmo dia
        FROM FILTRADOS_IMPORTANTES_DESAMBIGUADOS
        WHERE stao_tppr_dk IN {DENUNCIA}
        GROUP BY pip_codigo
        """.format(DENUNCIA=DENUNCIA)
    ).createOrReplaceTempView("DENUNCIA")

    spark.sql(
    """
    SELECT
	pip_codigo AS orgao_id,
        COUNT(DISTINCT docu_dk) as finalizacoes --distinct docu_dk para evitar andamentos duplicados no mesmo dia
    FROM FILTRADOS_IMPORTANTES_DESAMBIGUADOS
    WHERE tipo in ('arquivado', 'acordo', 'denunciado', 'cautelado')
    GROUP BY pip_codigo
    """
    ).createOrReplaceTempView("FINALIZADOS")

    spark.sql(
        """
        SELECT pip_codigo as orgao_id,
        COUNT(vist_dk) as vistas
        FROM FILTRADOS_SEM_ANDAMENTO
        WHERE vist_dt_fechamento_vista > cast(date_sub(current_timestamp(), 30) as timestamp)
        OR vist_dt_fechamento_vista IS NULL
        GROUP BY pip_codigo
        """.format(days_past_end)).createOrReplaceTempView("VISTA_30_DIAS")

    spark.sql(
        """
        SELECT
        pip_codigo as orgao_id,
        COUNT(DISTINCT docu_dk) as resolutividade
    FROM FILTRADOS_IMPORTANTES_DESAMBIGUADOS
    WHERE stao_tppr_dk IN {FINALIZACOES} --arquvidado part 2/2
    AND (vist_dt_fechamento_vista > cast(date_sub(current_timestamp(), 30) as timestamp)
    OR vist_dt_fechamento_vista IS NULL)
    GROUP BY pip_codigo
    """.format(FINALIZACOES=DENUNCIA+ARQUIVAMENTO+ACORDO)
    ).createOrReplaceTempView("RESOLUCOES")


    indicadores_sucesso = spark.sql(
        """
            SELECT
                g.orgao_id,
                (d.denuncias/g.vistas) AS indice,
                'p_elucidacoes' AS tipo
            FROM GRUPO g
            JOIN DENUNCIA d ON g.orgao_id = d.orgao_id
            UNION ALL
            SELECT
               f.orgao_id,
               f.finalizacoes / g.vistas AS indice,
               'p_finalizacoes' AS tipo
            FROM FINALIZADOS f
            JOIN grupo g ON f.orgao_id = g.orgao_id
            UNION ALL
            SELECT v30.orgao_id,
            res.resolutividade / v30.vistas,
            'p_resolutividade' AS tipo
            FROM VISTA_30_DIAS v30
            JOIN RESOLUCOES res ON v30.orgao_id = res.orgao_id
        """.format(schema_exadata_aux)
    )

    output_table_name = options["table_name"]
    table_name = "{0}.{1}".format(schema_exadata_aux, output_table_name)
    indicadores_sucesso.write.mode("overwrite").saveAsTable(
        "temp_table_{0}".format(output_table_name)
    )
    temp_table = spark.table("temp_table_{0}".format(output_table_name))

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_{0}".format(output_table_name))

    spark.catalog.clearCache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create table indices de sucesso das PIPs"
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
        "-c",
        "--daysPastStart",
        metavar="daysPastStart",
        type=int,
        default=540,
        help="",
    )
    parser.add_argument(
        "-f",
        "--daysPastEnd",
        metavar="daysPastEnd",
        type=int,
        default=180,
        help="",
    )
    parser.add_argument(
        "-t",
        "--tableName",
        metavar="tableName",
        type=str,
        default="tb_pip_indicadores_sucesso",
        help="",
    )

    args = parser.parse_args()

    options = {
        "schema_exadata": args.schemaExadata,
        "schema_exadata_aux": args.schemaExadataAux,
        "impala_host": args.impalaHost,
        "impala_port": args.impalaPort,
        "days_past_start": args.daysPastStart,
        "days_past_end": args.daysPastEnd,
        "table_name": args.tableName,
    }

    execute_process(options)
