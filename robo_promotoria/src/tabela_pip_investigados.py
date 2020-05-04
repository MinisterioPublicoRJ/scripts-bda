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


    investigados = spark.sql("""
        SELECT pess_nm_pessoa, pip_codigo, docu_dk, docu_dt_cadastro
        FROM {0}.mcpr_personagem
        JOIN {0}.mcpr_pessoa ON pess_dk = pers_pess_dk
        JOIN {0}.mcpr_documento ON docu_dk = pers_docu_dk
        JOIN {0}.mcpr_tp_personagem ON pers_tppe_dk = tppe_dk
        JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) P ON pip_codigo = docu_orgi_orga_dk_responsavel
        WHERE pers_tppe_dk IN (290, 7, 21, 317, 20, 14, 32, 345, 40, 5)
        AND docu_tpst_dk != 11
        AND docu_fsdc_dk = 1
    """.format(schema_exadata, schema_exadata_aux))
    investigados.registerTempTable('INVESTIGADOS')

    table = spark.sql("""
        SELECT pess_nm_pessoa, pip_codigo, COUNT(*) as nr_investigacoes
        FROM INVESTIGADOS
        GROUP BY pess_nm_pessoa, pip_codigo
    """)

    new_names = spark.sql("""
        SELECT pip_codigo, collect_list(pess_nm_pessoa) as nomes
        FROM INVESTIGADOS
        WHERE to_date(docu_dt_cadastro) = to_date(current_timestamp())
        GROUP BY pip_codigo
    """).collect()

    conn = Connection('bda1node05.pgj.rj.gov.br')
    t = conn.table('dev:pip_investigados_flags')
    for orgao in new_names:
        orgao_id = str(orgao['pip_codigo'])
        nomes = orgao['nomes']
        removed_rows = t.scan(
            row_prefix=orgao_id,
            filter=(
                # "SingleColumnValueFilter ('flags','is_removed',=,'regexstring:^True$') AND "
                "DependentColumnFilter ('flags', 'is_removed')"
            )
        )
        for row in removed_rows:
            if row[1]['identificacao:nm_personagem'].decode('utf-8') in nomes:
                t.delete(row[0])

    table_name = "{}.tb_pip_investigados".format(schema_exadata_aux)

    table.write.mode("overwrite").saveAsTable("temp_table_pip_investigados")
    temp_table = spark.table("temp_table_pip_investigados")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_pip_investigados")

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])


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
