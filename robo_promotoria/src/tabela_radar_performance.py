import argparse

import pyspark
from pyspark.sql import Window
from pyspark.sql.functions import max, col, count, concat_ws, collect_list

from utils import _update_impala_table


def execute_process(options):
    spark = (
        pyspark.sql.session.SparkSession.builder.appName("Radar")
        .enableHiveSupport()
        .getOrCreate()
    )

    spark.sql(
        """
        SELECT  d.docu_dk,
                d.docu_orgi_orga_dk_responsavel as orgao_id,
                a.pcao_dt_andamento,
                s.stao_tppr_dk
        FROM {schema}.mcpr_documento d
        join {schema}.mcpr_vista v on v.vist_docu_dk = d.docu_dk
        join {schema}.mcpr_andamento a on a.pcao_vist_dk = v.vist_dk
        join {schema}.mcpr_sub_andamento s on s.stao_pcao_dk = a.pcao_dk
        WHERE to_date(pcao_dt_andamento)
            > to_date(date_sub(current_timestamp(), {days_ago}))
        AND to_date(pcao_dt_andamento) <= to_date(current_timestamp())
        GROUP BY docu_dk, d.docu_orgi_orga_dk_responsavel,
            a.pcao_dt_andamento, s.stao_tppr_dk
    """.format(schema=options["schema_exadata"], days_ago=options["days_ago"])
    ).createOrReplaceTempView("andamentos")
    spark.catalog.cacheTable("andamentos")

    spark.sql(
        """
    select docu_dk, orgao_id, pcao_dt_andamento, stao_tppr_dk
    from andamentos
    where stao_tppr_dk in (7912,6548,6326,6681,6678,6645,6682,6680,6679,6644,
                           6668,6666,6665,6669,6667,6664,6655,6662,6659,6658,
                           6663,6661,6660,6657,6670,6676,6674,6673,6677,6675,
                           6672,6018,6341,6338,6019,6017,6591,6339,6553,7871,
                           6343,6340,6342,6021,6334,6331,6022,6020,6593,6332,
                           7872,6336,6333,6335,7745,6346,6345,6015,6016,6325,
                           6327,6328,6329,6330,6337,6344,6656,6671,7869,7870,
                           6324,6322,6011,6012,6013,1092,1094,1095, 6370,6251)
    """
    ).createOrReplaceTempView("andamentos_codigos")

    cancela_indeferimento = spark.sql(
        """
        select docu_dk, orgao_id, pcao_dt_andamento, -1 as tipo_andamento
        from andamentos where stao_tppr_dk = 6007
        """
    )

    documento_andamentos = spark.sql(
        """
        select
            docu_dk,
            orgao_id,
            CASE WHEN stao_tppr_dk IN (7912,6548,6681,6678,6645,6682,6680,
                                       6679,6644,6668,6666,6665,6669,6667,6664,
                                       6662,6659,6658,6663,6661,6660,6657,
                                       6670,6676,6674,6673,6677,6675,6672,6018,
                                       6341,6338,6019,6017,6591,6339,6553,7871,
                                       6343,6340,6342,6021,6334,6331,6022,6020,
                                       6593,6332,7872,6336,6333,6335,7745,6346,
                                       6345,6015,6016,6325,6327,6328,6329,6330,
                                       6337,6344,6656,6671,7869,7870,6324)
             THEN 1
             WHEN stao_tppr_dk = 6322 THEN 2
             WHEN stao_tppr_dk IN (6011, 6012, 6013, 1092, 1094, 1095) THEN 3
             WHEN stao_tppr_dk IN (6655, 6326, 6370) THEN 4
             WHEN stao_tppr_dk = 6251 THEN 5 end tipo_andamento,
            pcao_dt_andamento
        from andamentos_codigos
    """
    )

    cancela_df = cancela_indeferimento.groupby(
        ["orgao_id", "docu_dk", "pcao_dt_andamento"]
    ).agg(max("tipo_andamento").alias("tipo_andamento"))

    documento_df = documento_andamentos.groupby(
        ["orgao_id", "docu_dk", "pcao_dt_andamento"]
    ).agg(max("tipo_andamento").alias("tipo_andamento"))

    final_df = (
        documento_df.alias("d")
        .join(
            cancela_df.alias("c"),
            (col("d.docu_dk") == col("c.docu_dk"))
            & (col("c.pcao_dt_andamento") >= col("d.pcao_dt_andamento")),
            "left",
        )
        .where("c.tipo_andamento is null")
        .groupby(["d.orgao_id"])
        .pivot("d.tipo_andamento")
        .agg(count("d.tipo_andamento"))
        .na.fill(0)
        .withColumnRenamed("1", "arquivamento")
        .withColumnRenamed("2", "indeferimento")
        .withColumnRenamed("3", "instauracao")
        .withColumnRenamed("4", "tac")
        .withColumnRenamed("5", "acao")
    )

    final_df.createOrReplaceTempView("final_andamentos")
    spark.sql(
        """
            SELECT fa.*, ap.cod_pct, ap.pacote_atribuicao,
            ap.orgi_nm_orgao nm_orgao
            FROM final_andamentos fa
            INNER JOIN {schema_aux}.atualizacao_pj_pacote ap
            ON ap.id_orgao = fa.orgao_id
    """.format(
            schema_aux=options["schema_exadata_aux"]
        )
    ).createOrReplaceTempView("final_com_pacote")

    max_pacote = spark.sql(
        """
                   SELECT cod_pct, nm_orgao,
                   max(arquivamento) as max_arq,
                   max(indeferimento) as max_indef,
                   max(instauracao) as max_inst,
                   max(tac) as max_tac,
                   max(acao) as max_acoes
                   FROM
                   final_com_pacote fp
                   GROUP BY cod_pct, nm_orgao
    """
    )
    w = Window.partitionBy("cod_pct")
    orgao_max_arq = (
        max_pacote.withColumn("m_max_arq", max("max_arq").over(w))
        .where(col("max_arq") == col("m_max_arq"))
        .select(["cod_pct", "nm_orgao"])
        .groupBy("cod_pct")
        .agg(
            concat_ws(", ", collect_list("nm_orgao"))
            .alias("nm_max_arquivamentos")
        )
        .withColumnRenamed("cod_pct", "arq_cod_pct")
    )
    orgao_max_indef = (
        max_pacote.withColumn("m_max_indef", max("max_indef").over(w))
        .where(col("max_indef") == col("m_max_indef"))
        .select(["cod_pct", "nm_orgao"])
        .groupBy("cod_pct")
        .agg(
            concat_ws(", ", collect_list("nm_orgao"))
            .alias("nm_max_indeferimentos")
        )
        .withColumnRenamed("cod_pct", "indef_cod_pct")
    )
    orgao_max_inst = (
        max_pacote.withColumn("m_max_inst", max("max_inst").over(w))
        .where(col("max_inst") == col("m_max_inst"))
        .select(["cod_pct", "nm_orgao"])
        .groupBy("cod_pct")
        .agg(
            concat_ws(", ", collect_list("nm_orgao"))
            .alias("nm_max_instauracoes")
        )
        .withColumnRenamed("cod_pct", "inst_cod_pct")
    )
    orgao_max_tac = (
        max_pacote.withColumn("m_max_tac", max("max_tac").over(w))
        .where(col("max_tac") == col("m_max_tac"))
        .select(["cod_pct", "nm_orgao"])
        .groupBy("cod_pct")
        .agg(
            concat_ws(", ", collect_list("nm_orgao"))
            .alias("nm_max_tac")
        )
        .withColumnRenamed("cod_pct", "tac_cod_pct")
    )
    orgao_max_acoes = (
        max_pacote.withColumn("m_max_acoes", max("max_acoes").over(w))
        .where(col("max_acoes") == col("m_max_acoes"))
        .select(["cod_pct", "nm_orgao"])
        .groupBy("cod_pct")
        .agg(
            concat_ws(", ", collect_list("nm_orgao"))
            .alias("nm_max_acoes")
        )
        .withColumnRenamed("cod_pct", "acoes_cod_pct")
    )

    spark.sql(
        """
            SELECT cod_pct, max(arquivamento) as max_pacote_arquivamentos,
                   max(indeferimento) as max_pacote_indeferimentos,
                   max(instauracao) as max_pacote_instauracoes,
                   max(tac) as max_pacote_tac,
                   max(acao) as max_pacote_acoes,
                   percentile(arquivamento, 0.5) as med_pacote_arquivamentos,
                   percentile(indeferimento, 0.5) as med_pacote_indeferimentos,
                   percentile(instauracao, 0.5) as med_pacote_instauracoes,
                   percentile(tac, 0.5) as med_pacote_tac,
                   percentile(acao, 0.5) as med_pacote_acoes
                   FROM final_com_pacote
                   GROUP BY cod_pct
    """
    ).createOrReplaceTempView("stats_pacote")
    stats = (
        spark.sql(
            """
            SELECT fp.cod_pct,
                   fp.pacote_atribuicao,
                   fp.orgao_id,
                   arquivamento as nr_arquivamentos,
                   indeferimento as nr_indeferimentos,
                   instauracao as nr_instauracaoes,
                   tac as nr_tac,
                   acao as nr_acoes,
                   max_pacote_arquivamentos,
                   max_pacote_indeferimentos,
                   max_pacote_instauracoes,
                   max_pacote_tac,
                   max_pacote_acoes,
                   arquivamento / max_pacote_arquivamentos
                       as perc_arquivamentos,
                   indeferimento / max_pacote_indeferimentos
                       as perc_indeferimentos,
                   instauracao / max_pacote_instauracoes
                       as perc_instauracaoes,
                   tac / max_pacote_tac as perc_tac,
                   acao / max_pacote_acoes as perc_acoes,
                   med_pacote_arquivamentos,
                   med_pacote_indeferimentos,
                   med_pacote_instauracoes,
                   med_pacote_tac,
                   med_pacote_acoes,
                   (arquivamento - med_pacote_arquivamentos)
                       / med_pacote_arquivamentos as var_med_arquivaentos,
                   (indeferimento - med_pacote_indeferimentos)
                       / med_pacote_indeferimentos as var_med_indeferimentos,
                   (instauracao - med_pacote_instauracoes)
                       / med_pacote_instauracoes as var_med_instauracoes,
                   (tac - med_pacote_tac) / med_pacote_tac as var_med_tac,
                   (acao - med_pacote_acoes)
                       / med_pacote_acoes as var_med_acoes,
                   current_timestamp() as dt_calculo
            FROM final_com_pacote fp
            INNER JOIN stats_pacote sp
            ON fp.cod_pct = sp.cod_pct
    """
        )
        .join(orgao_max_arq, col("cod_pct") == col("arq_cod_pct"))
        .drop("arq_cod_pct")
        .join(orgao_max_indef, col("cod_pct") == col("indef_cod_pct"))
        .drop("indef_cod_pct")
        .join(orgao_max_inst, col("cod_pct") == col("inst_cod_pct"))
        .drop("inst_cod_pct")
        .join(orgao_max_tac, col("cod_pct") == col("tac_cod_pct"))
        .drop("tac_cod_pct")
        .join(orgao_max_acoes, col("cod_pct") == col("acoes_cod_pct"))
        .drop("acoes_cod_pct")
    )
    table_name = "{}.tb_radar_performance".format(
        options["schema_exadata_aux"]
    )

    stats.write.mode("overwrite").saveAsTable("temp_table_radar_performance")
    temp_table = spark.table("temp_table_radar_performance")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_radar_performance")

    _update_impala_table(
        table_name, options['impala_host'], options['impala_port']
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create table radar performance"
    )
    parser.add_argument(
        '-e',
        '--schemaExadata',
        metavar='schemaExadata',
        type=str,
        help=''
    )
    parser.add_argument(
        '-a',
        '--schemaExadataAux',
        metavar='schemaExadataAux',
        type=str,
        help=''
    )
    parser.add_argument(
        '-i',
        '--impalaHost',
        metavar='impalaHost',
        type=str,
        help=''
    )
    parser.add_argument(
        '-o',
        '--impalaPort',
        metavar='impalaPort',
        type=str,
        help=''
    )
    args = parser.parse_args()

    options = {
        'schema_exadata': args.schemaExadata,
        'schema_exadata_aux': args.schemaExadataAux,
        'impala_host': args.impalaHost,
        'impala_port': args.impalaPort,
        'days_ago': 180,
    }

    execute_process(options)
