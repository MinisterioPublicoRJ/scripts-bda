import pyspark
from utils import _update_impala_table
from happybase import Connection
import argparse


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_pip_investigados_representantes") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    PERS_DOCS_PIPS = spark.sql("""
        SELECT DISTINCT pers_pess_dk
        FROM {0}.mcpr_personagem
        JOIN {0}.mcpr_documento ON docu_dk = pers_docu_dk
        JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) P ON pip_codigo = docu_orgi_orga_dk_responsavel
        WHERE pers_tppe_dk IN (290, 7, 21, 317, 20, 14, 32, 345, 40, 5)
        AND docu_tpst_dk != 11
    """.format(schema_exadata, schema_exadata_aux))
    PERS_DOCS_PIPS.createOrReplaceTempView('PERS_DOCS_PIPS')
    spark.catalog.cacheTable('PERS_DOCS_PIPS')


    investigados_fisicos_pip_total = spark.sql("""
        SELECT pesf_pess_dk, pesf_nm_pessoa_fisica, pesf_cpf, pesf_nm_mae, pesf_dt_nasc
        FROM PERS_DOCS_PIPS
        JOIN {0}.mcpr_pessoa_fisica ON pers_pess_dk = pesf_pess_dk
    """.format(schema_exadata, schema_exadata_aux))
    investigados_fisicos_pip_total.createOrReplaceTempView("INVESTIGADOS_FISICOS_PIP_TOTAL")
    spark.catalog.cacheTable('INVESTIGADOS_FISICOS_PIP_TOTAL')

    investigados_juridicos_pip_total = spark.sql("""
        SELECT pesj_pess_dk, pesj_nm_pessoa_juridica, pesj_cnpj
        FROM PERS_DOCS_PIPS
        JOIN {0}.mcpr_pessoa_juridica ON pers_pess_dk = pesj_pess_dk
    """.format(schema_exadata, schema_exadata_aux))
    investigados_juridicos_pip_total.createOrReplaceTempView("INVESTIGADOS_JURIDICOS_PIP_TOTAL")
    spark.catalog.cacheTable('INVESTIGADOS_JURIDICOS_PIP_TOTAL')

    
    pessoas_fisicas_representativas_1 = spark.sql("""
        SELECT t.pess_dk, min(t.representante_dk) as representante_dk
        FROM (
            SELECT A.pesf_pess_dk as pess_dk, B.pesf_pess_dk as representante_dk
            FROM INVESTIGADOS_FISICOS_PIP_TOTAL A
            JOIN INVESTIGADOS_FISICOS_PIP_TOTAL B ON A.PESF_PESS_DK = B.PESF_PESS_DK
            UNION ALL
            SELECT A.pesf_pess_dk as pess_dk, B.pesf_pess_dk as representante_dk
            FROM INVESTIGADOS_FISICOS_PIP_TOTAL A
            JOIN INVESTIGADOS_FISICOS_PIP_TOTAL B ON A.PESF_CPF = B.PESF_CPF
            UNION ALL
            SELECT A.pesf_pess_dk as pess_dk, B.pesf_pess_dk as representante_dk
            FROM INVESTIGADOS_FISICOS_PIP_TOTAL A
            JOIN INVESTIGADOS_FISICOS_PIP_TOTAL B ON A.pesf_nm_pessoa_fisica = B.pesf_nm_pessoa_fisica
                AND A.pesf_nm_mae = B.pesf_nm_mae
            UNION ALL
            SELECT A.pesf_pess_dk as pess_dk, B.pesf_pess_dk as representante_dk
            FROM INVESTIGADOS_FISICOS_PIP_TOTAL A
            JOIN INVESTIGADOS_FISICOS_PIP_TOTAL B ON A.pesf_nm_pessoa_fisica = B.pesf_nm_pessoa_fisica
                AND A.pesf_dt_nasc = B.pesf_dt_nasc
        ) t
        GROUP BY t.pess_dk
    """)
    pessoas_juridicas_representativas_1 = spark.sql("""
        SELECT t.pess_dk, min(t.representante_dk) as representante_dk
        FROM (
            SELECT A.pesj_pess_dk as pess_dk, B.pesj_pess_dk as representante_dk
            FROM INVESTIGADOS_JURIDICOS_PIP_TOTAL B
            JOIN INVESTIGADOS_JURIDICOS_PIP_TOTAL A ON B.pesj_pess_dk = A.pesj_pess_dk
            UNION ALL
            SELECT A.pesj_pess_dk as pess_dk, B.pesj_pess_dk as representante_dk
            FROM INVESTIGADOS_JURIDICOS_PIP_TOTAL B
            JOIN INVESTIGADOS_JURIDICOS_PIP_TOTAL A ON B.pesj_cnpj = A.pesj_cnpj
        ) t
        GROUP BY t.pess_dk
    """)
    pessoas_fisicas_representativas_1.createOrReplaceTempView("REPR_FISICO_1")
    pessoas_juridicas_representativas_1.createOrReplaceTempView("REPR_JURIDICO_1")

    repr_1 = spark.sql("""
        SELECT * FROM REPR_FISICO_1
        UNION ALL
        SELECT * FROM REPR_JURIDICO_1
    """)
    repr_1.createOrReplaceTempView("REPR_1")

    # Se 1 e representante de 2, e 2 e representante de 3, entao 1 deve ser representante de 3
    pessoas_representativas_2 = spark.sql("""
        SELECT A.pess_dk, B.representante_dk
        FROM REPR_1 A
        JOIN REPR_1 B ON A.representante_dk = B.pess_dk
    """)

    table_name = "{}.tb_pip_investigados_representantes".format(schema_exadata_aux)
    pessoas_representativas_2.write.mode("overwrite").saveAsTable("temp_table_pip_investigados_representantes")
    temp_table = spark.table("temp_table_pip_investigados_representantes")
    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_pip_investigados_representantes")
    _update_impala_table(table_name, options['impala_host'], options['impala_port'])

    spark.catalog.clearCache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table tabela pip_investigados_representantes")
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