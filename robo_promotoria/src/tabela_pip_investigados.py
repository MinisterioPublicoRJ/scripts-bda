#-*-coding:utf-8-*-
from datetime import datetime, timedelta
import os
import pyspark
import argparse

from happybase_kerberos_patch import KerberosConnection

from generic_utils import execute_compute_stats



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

    assunto_delito = spark.sql("""
        SELECT H.id, CASE WHEN P.nome_delito IS NOT NULL THEN P.nome_delito ELSE H.hierarquia END AS assunto
        FROM {0}.mmps_assunto_docto H
        LEFT JOIN {0}.tb_penas_assuntos P ON P.id = H.id
    """.format(schema_exadata_aux))
    assunto_delito.createOrReplaceTempView('assunto_delito')

    assuntos = spark.sql("""
        SELECT asdo_docu_dk, concat_ws(' --- ', collect_list(assunto)) as assuntos
        FROM {0}.mcpr_assunto_documento
        JOIN assunto_delito ON id = asdo_assu_dk
        WHERE asdo_dt_fim IS NULL
        GROUP BY asdo_docu_dk
    """.format(schema_exadata))
    assuntos.createOrReplaceTempView('assuntos')

    representantes_investigados = spark.sql("""
        SELECT DISTINCT representante_dk
        FROM {1}.tb_pip_investigados_representantes
        JOIN {0}.mcpr_personagem ON pess_dk = pers_pess_dk
        WHERE pers_tppe_dk IN (290, 7, 21, 317, 20, 14, 32, 345, 40, 5, 24)
    """.format(schema_exadata, schema_exadata_aux)
    )
    representantes_investigados.createOrReplaceTempView('representantes_investigados')

    documentos_pips = spark.sql("""
        SELECT 
            R.representante_dk,
            R.pess_dk,
            concat_ws(', ', collect_list(tppe_descricao)) as tppe_descricao,
            concat_ws(', ', collect_list(cast(tppe_dk as int))) as tppe_dk,
            pers_docu_dk as docu_dk,
            MIN(CASE WHEN pers_dt_fim <= current_timestamp() THEN 'Data Fim Atingida' ELSE 'Ativo' END) AS status_personagem,
            concat_ws(', ', collect_list(pers_dk)) as pers_dk
        FROM {0}.mcpr_personagem
        JOIN {0}.mcpr_tp_personagem ON tppe_dk = pers_tppe_dk
        JOIN {1}.tb_pip_investigados_representantes R ON pers_pess_dk = R.pess_dk
        JOIN representantes_investigados RI ON RI.representante_dk = R.representante_dk
        GROUP BY R.representante_dk, R.pess_dk, pers_docu_dk
    """.format(schema_exadata, schema_exadata_aux))
    documentos_pips.createOrReplaceTempView('documentos_pips')
    spark.catalog.cacheTable('documentos_pips')

    documentos_investigados = spark.sql("""
        WITH docs_representantes AS (
            SELECT DISTINCT docu_dk, representante_dk
            FROM documentos_pips
        ),
        tb_coautores AS (
            SELECT A.docu_dk, A.representante_dk,
                concat_ws(', ', collect_list(C.pess_nm_pessoa)) as coautores
            FROM docs_representantes A
            JOIN docs_representantes B ON A.docu_dk = B.docu_dk AND A.representante_dk != B.representante_dk
            JOIN {0}.mcpr_pessoa C ON C.pess_dk = B.representante_dk
            GROUP BY A.docu_dk, A.representante_dk
        ),
        ultimos_andamentos AS (
            SELECT docu_dk, pcao_dt_andamento, tppr_descricao, row_number() over (partition by docu_dk order by pcao_dt_andamento desc) as nr_and
            FROM (SELECT DISTINCT docu_dk FROM documentos_pips) p
            LEFT JOIN {0}.mcpr_vista ON vist_docu_dk = docu_dk
            LEFT JOIN {0}.mcpr_andamento ON pcao_vist_dk = vist_dk
            LEFT JOIN {0}.mcpr_sub_andamento ON stao_pcao_dk = pcao_dk
            LEFT JOIN {0}.mcpr_tp_andamento ON stao_tppr_dk = tppr_dk
        )
        SELECT 
            D.representante_dk,
            coautores,
            tppe_descricao,
            tppe_dk,
            docu_orgi_orga_dk_responsavel as pip_codigo,
            D.docu_dk,
            DOCS.docu_nr_mp,
            DOCS.docu_nr_externo,
            DOCS.docu_dt_cadastro,
            cldc_ds_classe,
            O.orgi_nm_orgao,
            DOCS.docu_tx_etiqueta,
            assuntos,
            fsdc_ds_fase,
            pcao_dt_andamento as dt_ultimo_andamento,
            tppr_descricao as desc_ultimo_andamento,
            D.pess_dk,
            status_personagem,
            pers_dk,
            cod_pct,
            cast(substring(cast(D.representante_dk as string), -1, 1) as int) as rep_last_digit
        FROM documentos_pips D
        JOIN {0}.mcpr_documento DOCS ON DOCS.docu_dk = D.docu_dk
        LEFT JOIN tb_coautores CA ON CA.docu_dk = D.docu_dk AND CA.representante_dk = D.representante_dk
        LEFT JOIN (SELECT * FROM ultimos_andamentos WHERE nr_and = 1) UA ON UA.docu_dk = D.docu_dk
        JOIN {0}.orgi_orgao O ON orgi_dk = docu_orgi_orga_dk_responsavel
        LEFT JOIN {0}.mcpr_classe_docto_mp ON cldc_dk = docu_cldc_dk
        LEFT JOIN assuntos TASSU ON asdo_docu_dk = D.docu_dk
        LEFT JOIN {0}.mcpr_fases_documento ON docu_fsdc_dk = fsdc_dk
        LEFT JOIN {1}.atualizacao_pj_pacote ON id_orgao = docu_orgi_orga_dk_responsavel
        WHERE docu_tpst_dk != 11
    """.format(schema_exadata, schema_exadata_aux))

    table_name_procedimentos = options['table_name_procedimentos']
    table_name = "{}.{}".format(schema_exadata_aux, table_name_procedimentos)
    documentos_investigados.repartition("rep_last_digit").write.mode("overwrite").saveAsTable("temp_table_pip_investigados_procedimentos")
    temp_table = spark.table("temp_table_pip_investigados_procedimentos")
    temp_table.repartition(15).write.mode("overwrite").partitionBy('rep_last_digit').saveAsTable(table_name)
    spark.sql("drop table temp_table_pip_investigados_procedimentos")

    spark.catalog.clearCache()

    execute_compute_stats(table_name)

    # Contagem só será utilizada pela PIP e para PIPs, pelo menos por enquanto
    table = spark.sql("""
        WITH DOCS_INVESTIGADOS_FILTERED AS (
            SELECT representante_dk, pess_dk, docu_dk, docu_dt_cadastro, pip_codigo, fsdc_ds_fase
            FROM {1}.tb_pip_investigados_procedimentos
            WHERE tppe_dk REGEXP '(^| )(290|7|21|317|20|14|32|345|40|5|24)(,|$)' -- apenas os investigados na contagem
            AND cod_pct IN (200, 201, 202, 203, 204, 205, 206, 207, 208, 209)
        ),
        DISTINCT_DOCS_COUNT AS (
            SELECT representante_dk, COUNT(DISTINCT docu_dk) as nr_investigacoes, MAX(docu_dt_cadastro) as max_docu_date
            FROM DOCS_INVESTIGADOS_FILTERED
            JOIN {0}.mcpr_pessoa P ON P.pess_dk = representante_dk
            WHERE P.pess_nm_pessoa NOT REGEXP 'IDENTIFICADO|IGNORAD[OA]|APURA[CÇ][AÃ]O'
            GROUP BY representante_dk
        )
        SELECT 
            pess_nm_pessoa,
            t.*,
            MULTI.flag_multipromotoria,
            TOPN.flag_top50,
            cast(substring(cast(t.pip_codigo as string), -1, 1) as int) as orgao_last_digit
        FROM (
            SELECT c.representante_dk, pip_codigo, nr_investigacoes
            FROM DISTINCT_DOCS_COUNT c
            JOIN (
                SELECT representante_dk, pip_codigo
                FROM DOCS_INVESTIGADOS_FILTERED
                GROUP BY representante_dk, pip_codigo
                HAVING SUM(CASE WHEN fsdc_ds_fase = "Em Andamento" THEN 1 ELSE 0 END) > 0
            ) r
                ON c.representante_dk = r.representante_dk) t
        LEFT JOIN (
            SELECT representante_dk, True as flag_multipromotoria
            FROM DOCS_INVESTIGADOS_FILTERED
            GROUP BY representante_dk
            HAVING COUNT(DISTINCT pip_codigo) > 1
        ) MULTI ON MULTI.representante_dk = t.representante_dk
        LEFT JOIN (
            SELECT representante_dk, True as flag_top50
            FROM DISTINCT_DOCS_COUNT
            ORDER BY nr_investigacoes DESC, max_docu_date DESC
            LIMIT 50
        ) TOPN ON TOPN.representante_dk = t.representante_dk
        JOIN {0}.mcpr_pessoa ON pess_dk = t.representante_dk
    """.format(schema_exadata, schema_exadata_aux))

    table_name_investigados = options['table_name_investigados']
    table_name = "{}.{}".format(schema_exadata_aux, table_name_investigados)
    table.repartition('orgao_last_digit').write.mode("overwrite").saveAsTable("temp_table_pip_investigados")
    temp_table = spark.table("temp_table_pip_investigados")
    temp_table.repartition(10).write.mode("overwrite").partitionBy('orgao_last_digit').saveAsTable(table_name)
    spark.sql("drop table temp_table_pip_investigados")

    execute_compute_stats(table_name)


    # Investigados que aparecem em documentos novos reiniciam flags no HBase
    table_name_dt_checked = options['table_name_dt_checked']
    is_exists_dt_checked = check_table_exists(spark, schema_exadata_aux, table_name_dt_checked)
    current_time = datetime.now()

    if not is_exists_dt_checked:
        new_names = spark.sql("""
            SELECT pip_codigo, collect_list(representante_dk) as representantes
            FROM {0}.tb_pip_investigados_procedimentos
            WHERE docu_dt_cadastro > '{1}'
            GROUP BY pip_codigo
        """.format(schema_exadata_aux, str(current_time - timedelta(minutes=20)))).collect()
    else:
        new_names = spark.sql("""
            SELECT pip_codigo, collect_list(representante_dk) as representantes
            FROM {0}.tb_pip_investigados_procedimentos
            JOIN {0}.dt_checked_investigados
            WHERE docu_dt_cadastro > dt_ultima_verificacao
            GROUP BY pip_codigo
        """.format(schema_exadata_aux)).collect()

    conn = KerberosConnection(
            'bda1node05.pgj.rj.gov.br',
            timeout=3000,
            use_kerberos=True,
            protocol="compact",
        )
    #conn = Connection('bda1node05.pgj.rj.gov.br')
    t = conn.table('{}:pip_investigados_flags'.format(schema_hbase))
    for orgao in new_names:
        orgao_id = str(orgao['pip_codigo'])
        representantes = orgao['representantes']
        removed_rows = t.scan(
            row_prefix=orgao_id,
            filter=(
                "DependentColumnFilter ('flags', 'is_removed')"
            )
        )
        for row in removed_rows:
            if row[1].get("identificacao:representante_dk") and row[1]["identificacao:representante_dk"].decode('utf-8') in representantes:
                t.delete(row[0])

    # Usa tabela para guardar a data de ultima verificacao de novos documentos
    tb_ultima_verificacao = spark.sql("""
        SELECT '{0}' as dt_ultima_verificacao
    """.format(str(current_time)))

    table_name = "{}.{}".format(schema_exadata_aux, table_name_dt_checked)
    tb_ultima_verificacao.write.mode("overwrite").saveAsTable(table_name)

    execute_compute_stats(table_name)


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
