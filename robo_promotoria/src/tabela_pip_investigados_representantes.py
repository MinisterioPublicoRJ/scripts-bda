import pyspark
from utils import _update_impala_table
from happybase import Connection
import argparse

def check_table_exists(spark, schema, table_name):
    spark.sql("use %s" % schema)
    result_table_check = spark.sql("SHOW TABLES LIKE '%s'" % table_name).count()
    return True if result_table_check > 0 else False


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_pip_investigados_representantes") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    DOCS_PIPS = spark.sql("""
        SELECT pers_pess_dk, pip_codigo, docu_nr_mp, docu_dt_cadastro, cldc_ds_classe, orgi_nm_orgao, docu_fsdc_dk
        FROM {0}.mcpr_personagem
        JOIN {0}.mcpr_documento ON docu_dk = pers_docu_dk
        JOIN {0}.mcpr_tp_personagem ON pers_tppe_dk = tppe_dk
        JOIN {0}.orgi_orgao ON orgi_dk = docu_orgi_orga_dk_responsavel
        JOIN {0}.mcpr_classe_docto_mp ON cldc_dk = docu_cldc_dk
        JOIN (SELECT DISTINCT pip_codigo FROM {1}.tb_pip_aisp) P ON pip_codigo = docu_orgi_orga_dk_responsavel
        WHERE pers_tppe_dk IN (290, 7, 21, 317, 20, 14, 32, 345, 40, 5)
        AND docu_tpst_dk != 11
    """.format(schema_exadata, schema_exadata_aux))
    DOCS_PIPS.createOrReplaceTempView('DOCS_PIPS')
    DOCS_PIPS.cache()

    # TODO: Pessoas juridicas
    # Historico de pessoas fisicas investigadas nas PIPs
    investigados_fisicos_pip_total = spark.sql("""
        SELECT pesf_pess_dk, pesf_nm_pessoa_fisica, pesf_cpf, pesf_nm_mae, pesf_dt_nasc
        FROM DOCS_PIPS
        JOIN {0}.mcpr_pessoa_fisica ON pers_pess_dk = pesf_pess_dk
    """.format(schema_exadata, schema_exadata_aux))
    investigados_fisicos_pip_total.createOrReplaceTempView("INVESTIGADOS_FISICOS_PIP_TOTAL")
    investigados_fisicos_pip_total.cache()

    investigados_juridicos_pip_total = spark.sql("""
        SELECT DISTINCT pesj_pess_dk, pesj_nm_pessoa_juridica, pesj_cnpj
        FROM DOCS_PIPS
        JOIN {0}.mcpr_pessoa_juridica ON pers_pess_dk = pesj_pess_dk
    """.format(schema_exadata, schema_exadata_aux))
    investigados_juridicos_pip_total.createOrReplaceTempView("INVESTIGADOS_JURIDICOS_PIP_TOTAL")
    investigados_juridicos_pip_total.cache()

    is_exists_table_representantes = check_table_exists(spark, schema_exadata_aux, "tb_pip_investigados_representantes")

    # Novas pessoas fisicas investigadas na PIP que ainda nao tem representante
    if not is_exists_table_representantes:
        investigados_fisicos_pip_novos = spark.sql("""
            SELECT *
            FROM INVESTIGADOS_FISICOS_PIP_TOTAL
        """)
        investigados_juridicos_pip_novos = spark.sql("""
            SELECT *
            FROM INVESTIGADOS_JURIDICOS_PIP_TOTAL
        """)
    else:
        investigados_fisicos_pip_novos = spark.sql("""
            SELECT * 
            FROM INVESTIGADOS_FISICOS_PIP_TOTAL t
            WHERE t.pesf_pess_dk NOT IN (
                SELECT pess_dk FROM exadata_aux_dev.tb_pip_investigados_representantes
            )
        """)
        investigados_juridicos_pip_novos = spark.sql("""
            SELECT * 
            FROM INVESTIGADOS_JURIDICOS_PIP_TOTAL t
            WHERE t.pesj_pess_dk NOT IN (
                SELECT pess_dk FROM exadata_aux_dev.tb_pip_investigados_representantes
            )
        """)
    investigados_fisicos_pip_novos.coalesce(10).createOrReplaceTempView("INVESTIGADOS_FISICOS_PIP_NOVOS")
    investigados_juridicos_pip_novos.coalesce(10).createOrReplaceTempView("INVESTIGADOS_JURIDICOS_PIP_NOVOS")


    pessoas_fisicas_representativas_1 = spark.sql("""
        SELECT A.pesf_pess_dk as pess_dk, min(B.pesf_pess_dk) as representante_dk
        FROM INVESTIGADOS_FISICOS_PIP_NOVOS A
        JOIN INVESTIGADOS_FISICOS_PIP_TOTAL B ON A.PESF_PESS_DK >= B.PESF_PESS_DK
        AND 
            (A.PESF_PESS_DK = B.PESF_PESS_DK OR 
            A.pesf_cpf = B.pesf_cpf OR 
            (A.pesf_nm_pessoa_fisica = B.pesf_nm_pessoa_fisica
            AND(
                A.pesf_nm_mae = B.pesf_nm_mae
                OR A.pesf_dt_nasc = B.pesf_dt_nasc
            )))
        GROUP BY A.pesf_pess_dk
    """)
    pessoas_juridicas_representativas_1 = spark.sql("""
        SELECT A.pesj_pess_dk as pess_dk, min(B.pesj_pess_dk) as representante_dk
        FROM INVESTIGADOS_JURIDICOS_PIP_NOVOS A
        JOIN INVESTIGADOS_JURIDICOS_PIP_TOTAL B ON B.pesj_pess_dk = A.pesj_pess_dk OR B.pesj_cnpj = A.pesj_cnpj
        GROUP BY A.pesj_pess_dk
    """)
    pessoas_fisicas_representativas_1.createOrReplaceTempView("REPR_FISICO_1")
    pessoas_juridicas_representativas_1.createOrReplaceTempView("REPR_JURIDICO_1")

    repr_1 = spark.sql("""
        SELECT * FROM REPR_FISICO_1
        UNION
        SELECT * FROM REPR_JURIDICO_1
    """)
    repr_1.createOrReplaceTempView("REPR_1")

    # Se 1 e representante de 2, e 2 e representante de 3, entao 1 deve ser representante de 3
    if not is_exists_table_representantes:
        pessoas_representativas_2 = spark.sql("""
            SELECT A.pess_dk, B.representante_dk
            FROM REPR_1 A
            JOIN REPR_1 B ON A.representante_dk = B.pess_dk
        """)
    else:
        pessoas_representativas_2 = spark.sql("""
            SELECT A.pess_dk, B.representante_dk
            FROM REPR_1 A
            JOIN (
                SELECT * FROM REPR_1
                UNION ALL
                SELECT * FROM {0}.tb_pip_investigados_representantes) B 
            ON A.representante_dk = B.pess_dk
        """.format(schema_exadata_aux))

    table_name = "{}.tb_pip_investigados_representantes".format(schema_exadata_aux)
    if is_exists_table_representantes:
        pessoas_representativas_2.coalesce(1).write.mode("append").insertInto(table_name)
    else:
        pessoas_representativas_2.write.mode("overwrite").saveAsTable(table_name)
    _update_impala_table(table_name, options['impala_host'], options['impala_port'])

    investigados_juridicos_pip_total.unpersist()
    investigados_fisicos_pip_total.unpersist()
    DOCS_PIPS.unpersist()


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