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

    documentos_investigados_ativos = spark.sql("""
        SELECT representante_dk, pip_codigo, docu_nr_mp, docu_dt_cadastro, cldc_ds_classe, orgi_nm_orgao
        FROM {0}.mcpr_personagem
        JOIN {0}.mcpr_documento ON docu_dk = pers_docu_dk
        JOIN {0}.mcpr_tp_personagem ON pers_tppe_dk = tppe_dk
        JOIN {0}.orgi_orgao ON orgi_dk = docu_orgi_orga_dk_responsavel
        JOIN {0}.mcpr_classe_docto_mp ON cldc_dk = docu_cldc_dk
        JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) P ON pip_codigo = docu_orgi_orga_dk_responsavel
        JOIN {1}.tb_pip_investigados_representantes ON pers_pess_dk = pess_dk
        WHERE pers_tppe_dk IN (290, 7, 21, 317, 20, 14, 32, 345, 40, 5)
        AND docu_tpst_dk != 11
        AND docu_fsdc_dk = 1
    """.format(schema_exadata, schema_exadata_aux))

    table_name = "{}.tb_pip_investigados_procedimentos".format(schema_exadata_aux)
    documentos_investigados_ativos.write.mode("overwrite").saveAsTable("temp_table_pip_investigados_procedimentos")
    temp_table = spark.table("temp_table_pip_investigados_procedimentos")
    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_pip_investigados_procedimentos")
    _update_impala_table(table_name, options['impala_host'], options['impala_port'])

    table = spark.sql("""
        SELECT pess_nm_pessoa, t.*, MULTI.flag_multipromotoria, TOPN.flag_top50
        FROM (
            SELECT representante_dk, pip_codigo, COUNT(*) as nr_investigacoes
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
            ORDER BY COUNT(*) DESC, MAX(docu_dt_cadastro) DESC
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
