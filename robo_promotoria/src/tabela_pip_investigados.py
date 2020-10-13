from datetime import datetime, timedelta
import os
os.environ['PYTHON_EGG_CACHE'] = "/tmp"
os.environ['PYTHON_EGG_DIR']='/tmp'

import pyspark
from happybase_kerberos_patch import KerberosConnection
import argparse


def check_table_exists(spark, schema, table_name):
    spark.sql("use %s" % schema)
    result_table_check = spark.sql("SHOW TABLES LIKE '%s'" % table_name).count()
    return True if result_table_check > 0 else False


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_pip_investigados") \
            .enableHiveSupport() \
            .getOrCreate()
    # Para evitar problemas de memoria causados por broadcast
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']
    schema_hbase = options['schema_hbase']

    lista_pips = spark.sql("""
        SELECT DISTINCT pip_codigo FROM {0}.tb_pip_aisp
        UNION ALL
        SELECT DISTINCT pip_codigo_antigo AS pip_codigo FROM {0}.tb_pip_aisp
    """.format(schema_exadata_aux))
    lista_pips.createOrReplaceTempView('lista_pips')
    spark.catalog.cacheTable("lista_pips")

    # Pega os assuntos com dt_fim NULL ou maior que a data atual
    # Alem disso, do codigo do orgao atual ou antigo
    # Por isso a query foi destrinchada em 4 partes
    assuntos = spark.sql("""
        SELECT asdo_docu_dk, concat_ws(' --- ', collect_list(hierarquia)) as assuntos
        FROM {0}.mcpr_assunto_documento
        JOIN {1}.mmps_assunto_docto ON id = asdo_assu_dk
        JOIN {0}.mcpr_documento ON asdo_docu_dk = docu_dk
        JOIN lista_pips P ON pip_codigo = docu_orgi_orga_dk_responsavel
        WHERE asdo_dt_fim IS NULL
        GROUP BY asdo_docu_dk
        UNION ALL
        SELECT asdo_docu_dk, concat_ws(' --- ', collect_list(hierarquia)) as assuntos
        FROM {0}.mcpr_assunto_documento
        JOIN {1}.mmps_assunto_docto ON id = asdo_assu_dk
        JOIN {0}.mcpr_documento ON asdo_docu_dk = docu_dk
        JOIN lista_pips P ON pip_codigo = docu_orgi_orga_dk_responsavel
        WHERE asdo_dt_fim > current_timestamp()
        GROUP BY asdo_docu_dk
    """.format(schema_exadata, schema_exadata_aux))
    assuntos.createOrReplaceTempView('assuntos')

    documentos_pips = spark.sql("""
        SELECT representante_dk, R.pess_dk, tppe_descricao, pip_codigo, docu_dk, docu_nr_mp, docu_dt_cadastro, docu_cldc_dk, docu_fsdc_dk, docu_tx_etiqueta,
            pers_dt_fim < current_timestamp() AS status_personagem
            --CASE WHEN pers_dt_fim <= current_timestamp() THEN 'Data Fim Atingida' ELSE 'Ativo' END AS status_personagem
        FROM {0}.mcpr_personagem
        JOIN {0}.mcpr_tp_personagem ON tppe_dk = pers_tppe_dk
        JOIN {1}.tb_pip_investigados_representantes R ON pers_pess_dk = R.pess_dk
        JOIN {0}.mcpr_documento ON docu_dk = pers_docu_dk
        JOIN lista_pips P ON pip_codigo = docu_orgi_orga_dk_responsavel
        WHERE pers_tppe_dk IN (290, 7, 21, 317, 20, 14, 32, 345, 40, 5, 24)
        AND docu_tpst_dk != 11
    """.format(schema_exadata, schema_exadata_aux))
    documentos_pips.createOrReplaceTempView('documentos_pips')

    documentos_investigados = spark.sql("""
        WITH tb_coautores AS (
            SELECT A.docu_dk, A.representante_dk,
                concat_ws(', ', collect_list(C.pess_nm_pessoa)) as coautores
            FROM documentos_pips A
            JOIN documentos_pips B ON A.docu_dk = B.docu_dk AND A.representante_dk != B.representante_dk
            JOIN {0}.mcpr_pessoa C ON C.pess_dk = B.representante_dk
            GROUP BY A.docu_dk, A.representante_dk
        ),
        ultimos_andamentos AS (
<<<<<<< Updated upstream
            SELECT docu_nr_mp, pcao_dt_andamento, tppr_descricao, row_number() over (partition by docu_dk order by pcao_dt_andamento desc) as nr_and
            FROM (SELECT DISTINCT docu_nr_mp, docu_dk FROM documentos_pips) p
=======
            SELECT docu_dk, pcao_dt_andamento, tppr_descricao, row_number() over (partition by docu_dk order by pcao_dt_andamento desc) as nr_and
            FROM (SELECT DISTINCT docu_dk FROM documentos_pips) p
>>>>>>> Stashed changes
            LEFT JOIN {0}.mcpr_vista ON vist_docu_dk = docu_dk
            LEFT JOIN {0}.mcpr_andamento ON pcao_vist_dk = vist_dk
            LEFT JOIN {0}.mcpr_sub_andamento ON stao_pcao_dk = pcao_dk
            LEFT JOIN {0}.mcpr_tp_andamento ON stao_tppr_dk = tppr_dk
        )
        SELECT D.representante_dk, coautores, tppe_descricao, pip_codigo, D.docu_dk, D.docu_nr_mp, docu_dt_cadastro, cldc_ds_classe, orgi_nm_orgao, docu_tx_etiqueta, assuntos, fsdc_ds_fase, pcao_dt_andamento as dt_ultimo_andamento, tppr_descricao as desc_ultimo_andamento, D.pess_dk, status_personagem
        FROM documentos_pips D
        LEFT JOIN tb_coautores CA ON CA.docu_dk = D.docu_dk AND CA.representante_dk = D.representante_dk
        LEFT JOIN (SELECT * FROM ultimos_andamentos WHERE nr_and = 1) UA ON UA.docu_dk = D.docu_dk
        JOIN {0}.orgi_orgao ON orgi_dk = pip_codigo
        LEFT JOIN {0}.mcpr_classe_docto_mp ON cldc_dk = docu_cldc_dk
        LEFT JOIN assuntos TASSU ON asdo_docu_dk = D.docu_dk
        LEFT JOIN {0}.mcpr_fases_documento ON docu_fsdc_dk = fsdc_dk
    """.format(schema_exadata, schema_exadata_aux))

    table_name_procedimentos = options['table_name_procedimentos']
    table_name = "{}.test_{}".format(schema_exadata_aux, table_name_procedimentos)
    documentos_investigados.write.mode("overwrite").saveAsTable("temp_table_pip_investigados_procedimentos")
    temp_table = spark.table("temp_table_pip_investigados_procedimentos")
    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_pip_investigados_procedimentos")

    spark.catalog.clearCache()

    # table = spark.sql("""
    #     SELECT pess_nm_pessoa, t.*, MULTI.flag_multipromotoria, TOPN.flag_top50
    #     FROM (
    #         SELECT c.representante_dk, pip_codigo, nr_investigacoes
    #         FROM (
    #             SELECT representante_dk, COUNT(1) as nr_investigacoes
    #             FROM {1}.tb_pip_investigados_procedimentos
    #             JOIN {0}.mcpr_pessoa P ON P.pess_dk = representante_dk
    #             WHERE P.pess_nm_pessoa NOT REGEXP 'IDENTIFICADO|IGNORAD[OA]|P.BLICO|JUSTI.A P.BLICA'
    #             GROUP BY representante_dk
    #         ) c
    #         JOIN (
    #             SELECT representante_dk, pip_codigo
    #             FROM {1}.tb_pip_investigados_procedimentos
    #             GROUP BY representante_dk, pip_codigo
    #             HAVING SUM(CASE WHEN fsdc_ds_fase = "Em Andamento" THEN 1 ELSE 0 END) > 0
    #         ) r
    #             ON c.representante_dk = r.representante_dk) t
    #     LEFT JOIN (
    #         SELECT representante_dk, True as flag_multipromotoria
    #         FROM {1}.tb_pip_investigados_procedimentos
    #         GROUP BY representante_dk
    #         HAVING COUNT(DISTINCT pip_codigo) > 1
    #     ) MULTI ON MULTI.representante_dk = t.representante_dk
    #     LEFT JOIN (
    #         SELECT representante_dk, True as flag_top50
    #         FROM {1}.tb_pip_investigados_procedimentos
    #         GROUP BY representante_dk
    #         ORDER BY COUNT(1) DESC, MAX(docu_dt_cadastro) DESC
    #         LIMIT 50
    #     ) TOPN ON TOPN.representante_dk = t.representante_dk
    #     JOIN {0}.mcpr_pessoa ON pess_dk = t.representante_dk
    # """.format(schema_exadata, schema_exadata_aux))

    # table_name_investigados = options['table_name_investigados']
    # table_name = "{}.{}".format(schema_exadata_aux, table_name_investigados)
    # table.write.mode("overwrite").saveAsTable("temp_table_pip_investigados")
    # temp_table = spark.table("temp_table_pip_investigados")
    # temp_table.write.mode("overwrite").saveAsTable(table_name)
    # spark.sql("drop table temp_table_pip_investigados")


    # Investigados que aparecem em documentos novos reiniciam flags no HBase
    # table_name_dt_checked = options['table_name_dt_checked']
    # is_exists_dt_checked = check_table_exists(spark, schema_exadata_aux, table_name_dt_checked)
    # current_time = datetime.now()

    # if not is_exists_dt_checked:
    #     new_names = spark.sql("""
    #         SELECT pip_codigo, collect_list(representante_dk) as representantes
    #         FROM {0}.tb_pip_investigados_procedimentos
    #         WHERE docu_dt_cadastro > '{1}'
    #         GROUP BY pip_codigo
    #     """.format(schema_exadata_aux, str(current_time - timedelta(minutes=20)))).collect()
    # else:
    #     new_names = spark.sql("""
    #         SELECT pip_codigo, collect_list(representante_dk) as representantes
    #         FROM {0}.tb_pip_investigados_procedimentos
    #         JOIN {0}.dt_checked_investigados
    #         WHERE docu_dt_cadastro > dt_ultima_verificacao
    #         GROUP BY pip_codigo
    #     """.format(schema_exadata_aux)).collect()

    # conn = KerberosConnection(
    #         'bda1node05.pgj.rj.gov.br',
    #         timeout=3000,
    #         use_kerberos=True,
    #         protocol="compact",
    #     )
    # #conn = Connection('bda1node05.pgj.rj.gov.br')
    # t = conn.table('{}:pip_investigados_flags'.format(schema_hbase))
    # for orgao in new_names:
    #     orgao_id = str(orgao['pip_codigo'])
    #     representantes = orgao['representantes']
    #     removed_rows = t.scan(
    #         row_prefix=orgao_id,
    #         filter=(
    #             "DependentColumnFilter ('flags', 'is_removed')"
    #         )
    #     )
    #     for row in removed_rows:
    #         if row[1].get("identificacao:representante_dk") and row[1]["identificacao:representante_dk"].decode('utf-8') in representantes:
    #             t.delete(row[0])

    # # Usa tabela para guardar a data de ultima verificacao de novos documentos
    # tb_ultima_verificacao = spark.sql("""
    #     SELECT '{0}' as dt_ultima_verificacao
    # """.format(str(current_time)))

    # table_name = "{}.{}".format(schema_exadata_aux, table_name_dt_checked)
    # tb_ultima_verificacao.write.mode("overwrite").saveAsTable(table_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table tabela pip_investigados")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    parser.add_argument('-sb','--schemaHbase', metavar='schemaHbase', type=str, help='')
    parser.add_argument('-t1','--tableNameProcedimentos', metavar='tableNameProcedimentos', type=str, help='')
    parser.add_argument('-t2','--tableNameInvestigados', metavar='tableNameInvestigados', type=str, help='')
    parser.add_argument('-t3','--tableNameDtChecked', metavar='tableNameDtChecked', type=str, help='')
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort,
                    'schema_hbase' : args.schemaHbase,
                    'table_name_procedimentos' : args.tableNameProcedimentos,
                    'table_name_investigados' : args.tableNameInvestigados,
                    'table_name_dt_checked' : args.tableNameDtChecked,
                }

    execute_process(options)
