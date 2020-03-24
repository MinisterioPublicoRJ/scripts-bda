import pyspark

spark = pyspark.sql.session.SparkSession \
                .builder \
                .appName("Execute SQL") \
                .enableHiveSupport() \
                .getOrCreate()


def execute_process(options):
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
        and d.docu_cldc_dk in (18, 126, 127, 159, 175, 176, 177, 441)
        and sa.stao_tppr_dk IN (6374,6375,6376,6377,6378)
    """.format(
             schema=options["schema_exadata"],
             schema_aux=options["schema_exadata_aux"])
         )\
        .createOrReplaceTempView("tramitacao_acoes_tempo_1")
    spark.catalog.cacheTable("tramitacao_acoes_tempo_1")

    spark.sql("""
        select docu_dk, max(dt_finalizacao) max_dt_finalizacao
        from tramitacao_acoes_tempo_1
        group by docu_dk
    """).createOrReplaceTempView("max_tramitacao_tempo_1")

    spark.sql("""
        select t.*
        from tramitacao_acoes_tempo_1 t
        join max_tramitacao_tempo_1 m on t.docu_dk = m.docu_dk
        and t.dt_finalizacao = m.max_dt_finalizacao
    """).createOrReplaceTempView("tramitacao_tempo_1_final")

    spark.sql("""
        select  cod_pct,
                avg(tempo_tramitacao) media_pacote,
                min(tempo_tramitacao) minimo_pacote,
                max(tempo_tramitacao) maximo_pacote,
                percentile(tempo_tramitacao, 0.5) as mediana_pacote
        from tramitacao_tempo_1_final
        group by cod_pct
        """).createOrReplaceTempView("tramitacao_tempo_1_pct")

    spark.sql("""
        select cod_pct,
               id_orgao,
               avg(tempo_tramitacao) media_orgao,
               min(tempo_tramitacao) minimo_orgao,
               max(tempo_tramitacao) maximo_orgao,
               percentile(tempo_tramitacao, 0.5) as mediana_orgao
        from tramitacao_tempo_1_final
        group by id_orgao, cod_pct
        """).createOrReplaceTempView("tramitacao_tempo_1_orgao")

    spark.sql("""
        select p.cod_pct,
               o.id_orgao,
               p.media_pacote as media_pacote_t1,
               p.minimo_pacote as minimo_pacote_t1,
               p.maximo_pacote as maximo_pacote_t1,
               o.media_orgao as media_orgao_t1,
               o.minimo_orgao as minimo_orgao_t1,
               o.maximo_orgao as maximo_orgao_t1,
               o.mediana_orgao as mediana_orgao_t1
        from tramitacao_tempo_1_pct p
        join tramitacao_tempo_1_orgao o on p.cod_pct = o.cod_pct
    """).createOrReplaceTempView("final_tempo_1")

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
        (sa.stao_tppr_dk IN (6393) or
        (sa.stao_tppr_dk in (6383,6377,6378,6384,6374,6375,6376,6380,6381,6382)
         and datediff(current_timestamp(), a.pcao_dt_andamento) > 60))
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
        select p.cod_pct,
                o.id_orgao,
                p.media_pacote as media_pacote_t2,
                p.minimo_pacote as minimo_pacote_t2,
                p.maximo_pacote as maximo_pacote_t2,
                o.media_orgao as media_orgao_t2,
                o.minimo_orgao as minimo_orgao_t2,
                o.maximo_orgao as maximo_orgao_t2,
                o.mediana_orgao as mediana_orgao_t2
        from tramitacao_tempo_2_pct p
        join tramitacao_tempo_2_orgao o on p.cod_pct = o.cod_pct
    """).createOrReplaceTempView("final_tempo_2")

    spark.sql("""
        select t1.cod_pct,
               t1.id_orgao,
               t1.media_pacote_t1,
               t1.minimo_pacote_t1,
               t1.maximo_pacote_t1,
               t1.media_orgao_t1,
               t1.minimo_orgao_t1,
               t1.maximo_orgao_t1,
               t1.mediana_orgao_t1,
               t2.media_pacote_t2,
               t2.minimo_pacote_t2,
               t2.maximo_pacote_t2,
               t2.media_orgao_t2,
               t2.minimo_orgao_t2,
               t2.maximo_orgao_t2,
               t2.mediana_orgao_t2
        from final_tempo_1 t1
        join final_tempo_2 t2 on t1.cod_pct = t2.cod_pct
        and t1.id_orgao = t2.id_orgao
    """).createOrReplaceTempView("tramitacao_acoes_final")

    spark.sql("""
        select * from tramitacao_ic_final ic
        join tramitacao_acoes_final acoes on ic.id_orgao = acoes.id_orgao
    """)
