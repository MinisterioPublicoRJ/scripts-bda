import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from utils import _update_impala_table
import argparse

def create_regra(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, days_past=30):
    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    vistas = spark.sql("""
        SELECT 
            docu_dk,
            vist_dk, 
            CASE WHEN docu_dt_cadastro >= date_sub(current_timestamp(), {2}) THEN 1 ELSE 0 END AS is_instaurado_intervalo, 
            CASE WHEN vist_dt_abertura_vista >= date_sub(current_timestamp(), {2}) THEN 1 ELSE 0 END AS is_atual,
            vist_orgi_orga_dk,
            pesf_cpf,
            CASE WHEN stao_tppr_dk IN {6} THEN 1 ELSE 0 END AS is_aproveitamento
        FROM {0}.mcpr_documento
        JOIN {0}.mcpr_vista ON docu_dk = vist_docu_dk
        JOIN {0}.mcpr_andamento ON pcao_vist_dk = vist_dk
        JOIN {0}.mcpr_sub_andamento ON stao_pcao_dk = pcao_dk
        JOIN {1}.atualizacao_pj_pacote ON id_orgao = vist_orgi_orga_dk
        JOIN {0}.mcpr_pessoa_fisica ON vist_pesf_pess_dk_resp_andam = pesf_pess_dk
        WHERE vist_dt_abertura_vista >= date_sub(current_timestamp(), {3})
        AND vist_dt_abertura_vista <= current_timestamp()
        AND docu_cldc_dk IN {5}
        AND cod_pct IN {4}
    """.format(schema_exadata, schema_exadata_aux, days_past, days_past*2, pacotes, cldc_dks, tppr_dks))
    nm_table_vistas = "VISTAS_{}".format(nm_tipo)
    vistas.createOrReplaceTempView(nm_table_vistas)
    spark.catalog.cacheTable(nm_table_vistas)

    atuais = spark.sql("""
        SELECT
            vist_orgi_orga_dk,
            pesf_cpf,
            COUNT(DISTINCT docu_dk) as nr_documentos_distintos_atual,
            SUM(nr_aberturas_vista) as nr_aberturas_vista_atual,
            SUM(has_aproveitamento) as nr_aproveitamentos_atual,
            SUM(is_instaurado) AS nr_instaurados_atual
        FROM (
            SELECT 
                vist_orgi_orga_dk,
                pesf_cpf,
                docu_dk,
                COUNT(vist_dk) AS nr_aberturas_vista,
                MAX(is_aproveitamento) AS has_aproveitamento,
                MAX(is_instaurado_intervalo) AS is_instaurado
            FROM {0} t
            WHERE is_atual = 1
            GROUP BY vist_orgi_orga_dk, pesf_cpf, docu_dk) t
        GROUP BY vist_orgi_orga_dk, pesf_cpf
    """.format(nm_table_vistas))
    nm_table_atuais = "ATUAIS_{}".format(nm_tipo)
    atuais.createOrReplaceTempView(nm_table_atuais)

    anteriores = spark.sql("""
        SELECT
            vist_orgi_orga_dk,
            pesf_cpf,
            COUNT(DISTINCT docu_dk) as nr_documentos_distintos_anterior,
            SUM(nr_aberturas_vista) as nr_aberturas_vista_anterior,
            SUM(has_aproveitamento) as nr_aproveitamentos_anterior,
            SUM(is_instaurado) AS nr_instaurados_anterior
        FROM (
            SELECT 
                vist_orgi_orga_dk,
                pesf_cpf,
                docu_dk,
                COUNT(vist_dk) AS nr_aberturas_vista,
                MAX(is_aproveitamento) AS has_aproveitamento,
                MAX(is_instaurado_intervalo) AS is_instaurado
            FROM {0} t
            WHERE is_atual = 0
            GROUP BY vist_orgi_orga_dk, pesf_cpf, docu_dk) t
        GROUP BY vist_orgi_orga_dk, pesf_cpf
    """.format(nm_table_vistas))
    nm_table_anteriores = "ANTERIORES_{}".format(nm_tipo)
    anteriores.createOrReplaceTempView(nm_table_anteriores)

    table = spark.sql("""
        SELECT '{0}' as tipo_detalhe, {1} as intervalo, 
        at.*, an.nr_documentos_distintos_anterior, an.nr_aberturas_vista_anterior, an.nr_aproveitamentos_anterior, an.nr_instaurados_anterior,
        (at.nr_documentos_distintos_atual - an.nr_documentos_distintos_anterior)/an.nr_documentos_distintos_anterior as variacao_documentos_distintos,
        (at.nr_aberturas_vista_atual - an.nr_aberturas_vista_anterior)/an.nr_aberturas_vista_anterior as variacao_aberturas_vista,
        (at.nr_aproveitamentos_atual - an.nr_aproveitamentos_anterior)/an.nr_aproveitamentos_anterior as variacao_aproveitamentos,
        (at.nr_instaurados_atual - an.nr_instaurados_anterior)/an.nr_instaurados_anterior as variacao_instaurados
        FROM {2} at
        LEFT JOIN {3} an ON at.vist_orgi_orga_dk = an.vist_orgi_orga_dk
            AND at.pesf_cpf = an.pesf_cpf
    """.format(nm_tipo, days_past, nm_table_atuais, nm_table_anteriores))
    nm_table_final = "DETALHE_{}".format(nm_tipo)
    table.createOrReplaceTempView(nm_table_final)

    return nm_table_final


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_detalhe_documentos") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    nm_tipo = 'pip_inqueritos'
    days_past = 30
    pacotes = "(200)"
    cldc_dks = "(3, 494)"
    tppr_dks = ("(6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,"
                "6346,6350,6359,6392,6017,6018,6020,7745,6648,6649,6650,6651,6652,6653,6654,"
                "6038,6039,6040,6041,6042,6043,7815,7816,6620,6257,6258,7878,7877,6367,6368,6369,6370,1208,1030,6252,6253,1201,1202,6254)")
    nm_table_1 = create_regra(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, days_past)

    nm_tipo = 'pip_pics'
    cldc_dks = "(590)"
    nm_table_2 = create_regra(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, days_past)

    table = spark.sql("""
        SELECT * FROM {0}
        UNION ALL
        SELECT * FROM {1}
    """.format(nm_table_1, nm_table_2))


    table_name = "{}.tb_detalhe_documentos_orgao_cpf".format(schema_exadata_aux)

    table.write.mode("overwrite").saveAsTable("temp_table_detalhe_documentos_orgao_cpf")
    temp_table = spark.table("temp_table_detalhe_documentos_orgao_cpf")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_detalhe_documentos_orgao_cpf")

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table detalhe documentos")
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
