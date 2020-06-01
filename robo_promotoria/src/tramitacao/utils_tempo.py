def execute_process(spark, options, cldc_dk_lista, tppr_dk_lista, pacote_lista, nome_regra):
    classes = cldc_dk_lista
    andamentos = tppr_dk_lista
    pacotes = pacote_lista

    nome_regra_slug = nome_regra.replace(" ", "_")

    spark.sql("""
     SELECT d.docu_dk, ap.id_orgao,
            datediff(a.pcao_dt_andamento, d.docu_dt_cadastro) tempo_tramitacao,
            a.pcao_dt_andamento dt_finalizacao, ap.cod_pct
            FROM {schema}.mcpr_documento d
            JOIN {schema}.mcpr_vista v on d.docu_dk = v.vist_docu_dk
            JOIN {schema}.mcpr_andamento a on v.vist_dk = a.pcao_vist_dk
            JOIN {schema}.mcpr_sub_andamento sa on sa.stao_pcao_dk = a.pcao_dk
            JOIN {schema_aux}.atualizacao_pj_pacote ap on ap.id_orgao = d.DOCU_ORGI_ORGA_DK_RESPONSAVEL
            WHERE a.pcao_dt_andamento > add_months(current_timestamp(),-12)
            and d.docu_cldc_dk IN {classes}
            and sa.stao_tppr_dk IN {andamentos}
            and ap.cod_pct IN {pacotes}
            AND docu_tpst_dk != 11
            AND pcao_dt_cancelamento IS NULL
    GROUP BY d.docu_dk, ap.id_orgao, tempo_tramitacao, dt_finalizacao, ap.cod_pct
    """.format(
        schema=options["schema_exadata"],
        schema_aux=options["schema_exadata_aux"],
        classes=classes,
        andamentos=andamentos,
        pacotes=pacotes
    )).createOrReplaceTempView("tramitacao_{}".format(nome_regra_slug))
    spark.catalog.cacheTable("tramitacao_{}".format(nome_regra_slug))

    spark.sql("""
        select docu_dk, max(dt_finalizacao) max_dt_finalizacao
        from tramitacao_{}
        group by docu_dk
    """.format(nome_regra_slug)
    ).createOrReplaceTempView("max_tramitacao_{}".format(nome_regra_slug))

    spark.sql("""
        select t.*
        from tramitacao_{0} t
        join max_tramitacao_{0} m on t.docu_dk = m.docu_dk
        and t.dt_finalizacao = m.max_dt_finalizacao
    """.format(nome_regra_slug)
    ).createOrReplaceTempView("tramitacao_final_{}".format(nome_regra_slug))

    spark.sql("""
        select cod_pct,
                avg(tempo_tramitacao) media_pacote,
                min(tempo_tramitacao) minimo_pacote,
                max(tempo_tramitacao) maximo_pacote,
                percentile(tempo_tramitacao, 0.5) as mediana_pacote
        from tramitacao_final_{}
        group by cod_pct
        """.format(nome_regra_slug)
    ).createOrReplaceTempView("tramitacao_pct_{}".format(nome_regra_slug))

    spark.sql("""
        select cod_pct,
                id_orgao,
                avg(tempo_tramitacao) media_orgao,
                min(tempo_tramitacao) minimo_orgao,
                max(tempo_tramitacao) maximo_orgao,
                percentile(tempo_tramitacao, 0.5) as mediana_orgao
        from tramitacao_final_{}
        group by id_orgao, cod_pct
        """.format(nome_regra_slug)
        ).createOrReplaceTempView("tramitacao_orgao_{}".format(nome_regra_slug))

    spark.sql("""
        select id_orgao, '{0}' as nome_regra,
        media_orgao, minimo_orgao, maximo_orgao, mediana_orgao,
        media_pacote, minimo_pacote, maximo_pacote, mediana_pacote
        from tramitacao_pct_{0} p
        join tramitacao_orgao_{0} o on p.cod_pct = o.cod_pct
    """.format(nome_regra_slug)
    ).createOrReplaceTempView("tramitacao_{}_final".format(nome_regra_slug))

    return "tramitacao_{}_final".format(nome_regra_slug)
