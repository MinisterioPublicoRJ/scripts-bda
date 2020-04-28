import pyspark
from utils import _update_impala_table
import argparse


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_pip_investigados") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    table = spark.sql("""
        SELECT pess_nm_pessoa, pip_codigo, COUNT(*) as nr_investigacoes
        FROM {0}.mcpr_personagem
        JOIN {0}.mcpr_pessoa ON pess_dk = pers_pess_dk
        JOIN {0}.mcpr_documento ON docu_dk = pers_docu_dk
        JOIN {1}.tb_pip_aisp ON pip_codigo = docu_orgi_orga_dk_responsavel
        WHERE pers_tppe_dk IN (290, 7, 21, 317, 20, 14, 32, 345, 40, 5)
        AND docu_tpst_dk != 11
        GROUP BY pess_nm_pessoa, pip_codigo
    """.format(schema_exadata, schema_exadata_aux))

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
