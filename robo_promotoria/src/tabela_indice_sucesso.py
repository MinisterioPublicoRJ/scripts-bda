import argparse

import pyspark


def execute_process(options):

    spark = (
        pyspark.sql.session.SparkSession.builder.appName(
            "criar_tabela_indice_sucesso"
        )
        .enableHiveSupport()
        .getOrCreate()
    )

    schema_exadata = options["schema_exadata"]
    schema_exadata_aux = options["schema_exadata_aux"]

    days_past_start = options["days_past_start"]
    days_past_end = options["days_past_end"]
    spark.sql(
        """
        SELECT
        docu_dk,
        docu_nr_mp,
        pip_codigo,
        vist_dk,
        vist_docu_dk,
        vist_dt_abertura_vista,
        stao_tppr_dk,
        pcao_dt_andamento
        FROM {0}.mcpr_documento
        JOIN {0}.mcpr_vista ON vist_docu_dk = docu_dk
        JOIN (
                SELECT pip_codigo_antigo, pip_codigo
                from {1}.temp_pip_aisp
                GROUP BY pip_codigo_antigo, pip_codigo
            ) p
            ON p.pip_codigo_antigo = vist_orgi_orga_dk
            OR p.pip_codigo = vist_orgi_orga_dk
            OR p.pip_codigo_antigo = vist_orgi_orga_dk
        JOIN {0}.mcpr_pessoa_fisica pess ON pess.pesf_pess_dk = vist_pesf_pess_dk_resp_andam
        JOIN {0}.rh_funcionario f ON pess.pesf_cpf = f.cpf
        JOIN {0}.mcpr_andamento ON pcao_vist_dk = vist_dk
        JOIN {0}.mcpr_sub_andamento ON stao_pcao_dk = pcao_dk
        WHERE docu_cldc_dk IN (3, 494, 590) -- PIC e Inqueritos
        AND vist_dt_abertura_vista >= cast(date_sub(current_timestamp(), {2}) as timestamp)
        AND f.cdtipfunc IN ('1', '2')
    """.format(
        schema_exadata, schema_exadata_aux, days_past_start
    )).createOrReplaceTempView(
        "FILTRADOS"
    )

    spark.sql(
        """
        SELECT pip_codigo as orgao_id,
        COUNT(DISTINCT vist_docu_dk) as vistas
        FROM FILTRADOS
        WHERE vist_dt_abertura_vista <= cast(date_sub(current_timestamp(), {0}) as timestamp)
        GROUP BY pip_codigo
        """.format(days_past_end)).createOrReplaceTempView("grupo")

    spark.sql(
        """
        SELECT
            pip_codigo as orgao_id,
            count(Distinct vist_docu_dk) as denuncias
        FROM FILTRADOS
        WHERE stao_tppr_dk IN (6252, 6253, 1201, 1202, 6254)
        GROUP BY pip_codigo
        """.format(schema_exadata)
    ).createOrReplaceTempView("denuncia")

    spark.sql(
    """
     WITH TIPO_ANDAMENTO AS (
        SELECT
        pip_codigo,
        docu_dk,
        pcao_dt_andamento,
        CASE
            WHEN stao_tppr_dk in (6252, 6253, 1201, 1202, 6254) THEN 'denunciado'
            WHEN stao_tppr_dk in (7914, 7928, 7883, 7827) THEN 'acordado'
            WHEN stao_tppr_dk = 7920 THEN 'desacordado'
            WHEN stao_tppr_dk in (6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,6346,6350,6359,6392,6017,6018,6020,7745) THEN 'arquivado'
            WHEN stao_tppr_dk in (6075,1028,6798,7245,6307,1027,7803,6003,7802,7801) THEN 'desarquivado'
        END as tipo
        FROM FILTRADOS
    )
    SELECT
	TA.pip_codigo AS orgao_id,
	COUNT(TA.docu_dk) AS finalizacoes
    FROM TIPO_ANDAMENTO TA
    JOIN (SELECT pip_codigo, docu_dk, MAX(pcao_dt_andamento) AS ultimo_andamento FROM TIPO_ANDAMENTO GROUP BY pip_codigo, docu_dk) SUB_TA
    ON TA.pip_codigo = SUB_TA.pip_codigo AND TA.docu_dk = SUB_TA.docu_dk AND TA.pcao_dt_andamento = SUB_TA.ultimo_andamento
    WHERE TA.tipo in ('arquivado', 'acordo', 'denunciado')
    GROUP BY TA.pip_codigo
    """
    ).createOrReplaceTempView("FINALIZADOS")

    spark.sql(
        """
            SELECT
                g.orgao_id,
                (d.denuncias/g.vistas) AS indice,
		'p_elucidacoes' AS tipo
            FROM grupo g
            JOIN denuncia d ON g.orgao_id = d.orgao_id
	    UNION
            SELECT
		f.orgao_id,
		f.finalizacoes / g.vistas AS indice,
		'p_finalizacoes' AS tipo
	   FROM FINALIZADOS f
	   JOIN grupo g ON f.orgao_id = g.orgao_id
        """
    ).createOrReplaceTempView("INDICES_SUCESSO")
    table_name = "{}.tb_indicadores_sucesso".format(schema_exadata_aux)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create table distribuicao entradas"
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

    args = parser.parse_args()

    options = {
        "schema_exadata": args.schemaExadata,
        "schema_exadata_aux": args.schemaExadataAux,
        "impala_host": args.impalaHost,
        "impala_port": args.impalaPort,
        "days_past_start": args.daysPastStart,
        "days_past_end": args.daysPastEnd,
    }

    execute_process(options)
