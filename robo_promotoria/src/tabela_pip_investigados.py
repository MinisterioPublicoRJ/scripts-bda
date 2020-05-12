import pyspark
from utils import _update_impala_table
from happybase import Connection
import argparse


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_pip_investigados") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    lista_pips = spark.sql("""
        SELECT DISTINCT pip_codigo FROM {0}.tb_pip_aisp
    """.format(schema_exadata_aux))
    lista_pips.createOrReplaceTempView('lista_pips')
    spark.catalog.cacheTable("lista_pips")

    assuntos = spark.sql("""
        SELECT asdo_docu_dk, concat_ws(' --- ', collect_list(assu_descricao)) as assuntos
        FROM exadata_dev.mcpr_assunto_documento
        JOIN exadata_dev.mcpr_assunto ON assu_dk = asdo_assu_dk
        JOIN exadata_dev.mcpr_documento ON asdo_docu_dk = docu_dk
        JOIN lista_pips P ON pip_codigo = docu_orgi_orga_dk_responsavel
        WHERE asdo_dt_fim IS NULL
        GROUP BY asdo_docu_dk
        UNION ALL
        SELECT asdo_docu_dk, concat_ws(' --- ', collect_list(assu_descricao)) as assuntos
        FROM exadata_dev.mcpr_assunto_documento
        JOIN exadata_dev.mcpr_assunto ON assu_dk = asdo_assu_dk
        JOIN exadata_dev.mcpr_documento ON asdo_docu_dk = docu_dk
        JOIN lista_pips P ON pip_codigo = docu_orgi_orga_dk_responsavel
        WHERE asdo_dt_fim > current_timestamp()
        GROUP BY asdo_docu_dk
    """)
    assuntos.createOrReplaceTempView('assuntos')
    spark.catalog.cacheTable('assuntos')

    documentos_investigados_ativos = spark.sql("""
        SELECT representante_dk, pip_codigo, docu_nr_mp, docu_dt_cadastro, cldc_ds_classe, orgi_nm_orgao, docu_tx_etiqueta, assuntos
        FROM {0}.mcpr_personagem
        JOIN {1}.tb_pip_investigados_representantes ON pers_pess_dk = pess_dk
        JOIN {0}.mcpr_documento ON docu_dk = pers_docu_dk
        JOIN lista_pips P ON pip_codigo = docu_orgi_orga_dk_responsavel
        JOIN {0}.orgi_orgao ON orgi_dk = docu_orgi_orga_dk_responsavel
        JOIN {0}.mcpr_classe_docto_mp ON cldc_dk = docu_cldc_dk
        LEFT JOIN assuntos TASSU ON asdo_docu_dk = docu_dk 
        WHERE pers_tppe_dk IN (290, 7, 21, 317, 20, 14, 32, 345, 40, 5)
        AND docu_tpst_dk != 11
        AND docu_fsdc_dk = 1
        AND pers_dt_fim IS NULL OR pers_dt_fim > current_timestamp()
    """.format(schema_exadata, schema_exadata_aux))

    table_name = "{}.tb_pip_investigados_procedimentos".format(schema_exadata_aux)
    documentos_investigados_ativos.write.mode("overwrite").saveAsTable("temp_table_pip_investigados_procedimentos")
    temp_table = spark.table("temp_table_pip_investigados_procedimentos")
    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_pip_investigados_procedimentos")
    _update_impala_table(table_name, options['impala_host'], options['impala_port'])

    spark.catalog.clearCache()

    table = spark.sql("""
        SELECT pess_nm_pessoa, t.*, MULTI.flag_multipromotoria, TOPN.flag_top50
        FROM (
            SELECT representante_dk, pip_codigo, COUNT(1) as nr_investigacoes
            FROM {1}.tb_pip_investigados_procedimentos
            GROUP BY representante_dk, pip_codigo) t
        LEFT JOIN (
            SELECT representante_dk, True as flag_multipromotoria
            FROM exadata_aux_dev.tb_pip_investigados_procedimentos
            GROUP BY representante_dk
            HAVING COUNT(DISTINCT pip_codigo) > 1
        ) MULTI ON MULTI.representante_dk = t.representante_dk
        LEFT JOIN (
            SELECT representante_dk, True as flag_top50
            FROM exadata_aux_dev.tb_pip_investigados_procedimentos
            GROUP BY representante_dk
            ORDER BY COUNT(1) DESC, MAX(docu_dt_cadastro) DESC
            LIMIT 50
        ) TOPN ON TOPN.representante_dk = t.representante_dk
        JOIN {0}.mcpr_pessoa ON pess_dk = t.representante_dk
    """.format(schema_exadata, schema_exadata_aux))

    table_name = "{}.tb_pip_investigados".format(schema_exadata_aux)
    table.write.mode("overwrite").saveAsTable("temp_table_pip_investigados")
    temp_table = spark.table("temp_table_pip_investigados")
    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_pip_investigados")
    _update_impala_table(table_name, options['impala_host'], options['impala_port'])

    # Investigados que apareceram em documentos novos 
    # TODO: Verificar data e hora?
    # OPTIMIZATION: Possivel fazer sem collect()?
    new_names = spark.sql("""
        SELECT pip_codigo, collect_list(representante_dk) as representantes
        FROM {0}.tb_pip_investigados_procedimentos
        WHERE to_date(docu_dt_cadastro) = to_date(current_timestamp())
        GROUP BY pip_codigo
    """.format(schema_exadata_aux)).collect()

    conn = Connection('bda1node05.pgj.rj.gov.br')
    t = conn.table('dev:pip_investigados_flags')
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
            if row[1]['identificacao:representante_dk'].decode('utf-8') in representantes:
                t.delete(row[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table tabela pip_investigados")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort
                }

    execute_process(options)
