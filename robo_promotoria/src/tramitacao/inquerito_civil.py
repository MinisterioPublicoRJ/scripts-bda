def execute_process(spark, options):
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
            and d.docu_cldc_dk = 392
            and sa.stao_tppr_dk IN (7912,6548,6326,6681,6678,6645,6682,6680,
                                    6679,6644,6668,6666,6665,6669,6667,6664,
                                    6655,6662,6659,6658,6663,6661,6660,6657,
                                    6670,6676,6674,6673,6677,6675,6672,6018,
                                    6341,6338,6019,6017,6591,6339,6553,7871,
                                    6343,6340,6342,6021,6334,6331,6022,6020,
                                    6593,6332,7872,6336,6333,6335,7745,6346,
                                    6345,6015,6016,6325,6327,6328,6329,6330,
                                    6337,6344,6656,6671,7869,7870,6324,6251)
    GROUP BY d.docu_dk, ap.id_orgao, tempo_tramitacao, dt_finalizacao, ap.cod_pct
    """.format(
        schema=options["schema_exadata"],
        schema_aux=options["schema_exadata_aux"]
    )).createOrReplaceTempView("tramitacao_ic")
    spark.catalog.cacheTable("tramitacao_ic")

    spark.sql("""
        select docu_dk, max(dt_finalizacao) max_dt_finalizacao
        from tramitacao_ic
        group by docu_dk
    """).createOrReplaceTempView("max_tramitacao")

    spark.sql("""
        select t.*
        from tramitacao_ic t
        join max_tramitacao m on t.docu_dk = m.docu_dk
        and t.dt_finalizacao = m.max_dt_finalizacao
    """).createOrReplaceTempView("tramitacao_final")

    spark.sql("""
        select cod_pct,
                avg(tempo_tramitacao) media_pacote,
                min(tempo_tramitacao) minimo_pacote,
                max(tempo_tramitacao) maximo_pacote,
                percentile(tempo_tramitacao, 0.5) as mediana_pacote
        from tramitacao_final
        group by cod_pct
        """).createOrReplaceTempView("tramitacao_pct")

    spark.sql("""
        select cod_pct,
                id_orgao,
                avg(tempo_tramitacao) media_orgao,
                min(tempo_tramitacao) minimo_orgao,
                max(tempo_tramitacao) maximo_orgao,
                percentile(tempo_tramitacao, 0.5) as mediana_orgao
        from tramitacao_final
        group by id_orgao, cod_pct
        """).createOrReplaceTempView("tramitacao_orgao")

    spark.sql("""
        select *
        from tramitacao_pct p
        join tramitacao_orgao o on p.cod_pct = o.cod_pct
    """).createOrReplaceTempView("tramitacao_ic_final")
