def execute_process(spark, options):
    spark.sql("""
        SELECT
        d.docu_dk,
        ap.id_orgao,
        datediff(a.pcao_dt_andamento, d.docu_dt_cadastro) tempo_tramitacao,
        a.pcao_dt_andamento dt_finalizacao,
        ap.cod_pct
        from {schema}.mcpr_documento d
        join {schema}.mcpr_vista v
        on d.docu_dk = v.vist_docu_dk
        join {schema}.mcpr_andamento a
        on v.vist_dk = a.pcao_vist_dk
        join {schema}.mcpr_sub_andamento sa
        on sa.stao_pcao_dk = a.pcao_dk
        join {schema_aux}.atualizacao_pj_pacote ap
        on ap.id_orgao = d.DOCU_ORGI_ORGA_DK_RESPONSAVEL
        where
        a.pcao_dt_andamento > add_months(current_timestamp(),-12)
        and d.docu_cldc_dk in (18, 126, 127, 159, 175, 176, 177, 441) and
        (sa.stao_tppr_dk IN (6393, 7811) or
        (sa.stao_tppr_dk in (6383,6377,6378,6384,6374,6375,6376,6380,6381,6382)
         and datediff(current_timestamp(), a.pcao_dt_andamento) > 60))
        AND docu_tpst_dk != 11
        AND pcao_dt_cancelamento IS NULL
        AND ap.cod_pct IN (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)
         GROUP BY d.docu_dk, ap.id_orgao, tempo_tramitacao, dt_finalizacao, ap.cod_pct
    """.format(
             schema=options["schema_exadata"],
             schema_aux=options["schema_exadata_aux"])
        )\
        .createOrReplaceTempView("tramitacao_acoes_tempo_2")
    spark.catalog.cacheTable("tramitacao_acoes_tempo_2")

    spark.sql("""
        select docu_dk, max(dt_finalizacao) max_dt_finalizacao
        from tramitacao_acoes_tempo_2
        group by docu_dk
    """).createOrReplaceTempView("max_tramitacao_tempo_2")

    spark.sql("""
        select t.*
        from tramitacao_acoes_tempo_2 t
        join max_tramitacao_tempo_2 m on t.docu_dk = m.docu_dk
        and t.dt_finalizacao = m.max_dt_finalizacao
    """).createOrReplaceTempView("tramitacao_tempo_2_final")

    spark.sql("""
        select cod_pct,
               avg(tempo_tramitacao) media_pacote,
               min(tempo_tramitacao) minimo_pacote,
               max(tempo_tramitacao) maximo_pacote,
               percentile(tempo_tramitacao, 0.5) as mediana_pacote
        from tramitacao_tempo_2_final
        group by cod_pct
        """).createOrReplaceTempView("tramitacao_tempo_2_pct")

    spark.sql("""
        select cod_pct,
               id_orgao,
               avg(tempo_tramitacao) media_orgao,
               min(tempo_tramitacao) minimo_orgao,
               max(tempo_tramitacao) maximo_orgao,
               percentile(tempo_tramitacao, 0.5) as mediana_orgao
        from tramitacao_tempo_2_final
        group by id_orgao, cod_pct
        """).createOrReplaceTempView("tramitacao_tempo_2_orgao")

    spark.sql("""
        select o.id_orgao, 'tutela_acoes_transito_julgado' as nome_regra,
                o.media_orgao,
                o.minimo_orgao,
                o.maximo_orgao,
                o.mediana_orgao,
                p.media_pacote,
                p.minimo_pacote,
                p.maximo_pacote,
                p.mediana_pacote
        from tramitacao_tempo_2_pct p
        join tramitacao_tempo_2_orgao o on p.cod_pct = o.cod_pct
    """).createOrReplaceTempView("tutela_final_acoes_tempo_2")