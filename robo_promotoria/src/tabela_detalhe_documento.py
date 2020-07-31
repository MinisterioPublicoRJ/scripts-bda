import argparse
from datetime import timedelta, date

import pyspark

from utils import _update_impala_table
from detalhe_documento.utils_detalhes import create_regra_orgao, create_regra_cpf, setup_table_cache


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_detalhe_documentos") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    # Calcula datas para o mes corrente e mes anterior ate a mesma data (ou mais proxima)
    date_today = date.today()
    # Subtrair 30 a partir do dia 15 garante que caira sempre no mes anterior
    mid_last_month = (date_today.replace(day=15) - timedelta(30))

    date_mes_old_begin = mid_last_month.strftime('%Y-%m-01')
    date_mes_current_begin = date_today.strftime('%Y-%m-01')

    day = date_today.day
    while True:
        try:
            date_mes_old_end = mid_last_month.replace(day=day).strftime('%Y-%m-%d')
            break
        except:
            # Ocorre caso mes atual tenha mais dias que anterior
            day -= 1

    main_table = setup_table_cache(spark, options, date_mes_old_begin)

    # Tabela agregada orgao cpf
    ## Regras PIPs
    pacotes = "(200)"
    nm_intervalo = 'mes'
    nm_tipo = 'pip_inqueritos'
    cldc_dks = "(3, 494)"
    tppr_dks = ("(6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,"
                "6346,6350,6359,6392,6017,6018,6020,7745,6648,6649,6650,6651,6652,6653,6654,"
                "6038,6039,6040,6041,6042,6043,7815,7816,6620,6257,6258,7878,7877,6367,6368,6369,6370,1208,1030,6252,6253,1201,1202,6254)")
    nm_table_1 = create_regra_cpf(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_mes_old_begin,
                                  date_mes_old_end, date_mes_current_begin, nm_intervalo, vistas_table=main_table)

    nm_tipo = 'pip_pics'
    cldc_dks = "(590)"
    nm_table_2 = create_regra_cpf(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_mes_old_begin,
                                  date_mes_old_end, date_mes_current_begin, nm_intervalo, vistas_table=main_table)

    table_cpf = spark.sql("""
        SELECT * FROM {0}
        UNION ALL
        SELECT * FROM {1}
    """.format(nm_table_1, nm_table_2))

    table_name = "{}.tb_detalhe_documentos_orgao_cpf".format(schema_exadata_aux)
    table_cpf.write.mode("overwrite").saveAsTable("temp_table_detalhe_documentos_orgao_cpf")
    temp_table = spark.table("temp_table_detalhe_documentos_orgao_cpf")
    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_detalhe_documentos_orgao_cpf")

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])


    # Tabela agregada orgao
    ## Regras PIPs
    pacotes = "(200)"
    nm_intervalo = 'mes'
    nm_tipo = 'pip_inqueritos'
    cldc_dks = "(3, 494)"
    tppr_dks = ("(6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,"
                "6346,6350,6359,6392,6017,6018,6020,7745,6648,6649,6650,6651,6652,6653,6654,"
                "6038,6039,6040,6041,6042,6043,7815,7816,6620,6257,6258,7878,7877,6367,6368,6369,6370,1208,1030,6252,6253,1201,1202,6254)")
    nm_table_1 = create_regra_orgao(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_mes_old_begin,
                                    date_mes_old_end, date_mes_current_begin, nm_intervalo, vistas_table=main_table)

    nm_tipo = 'pip_pics'
    cldc_dks = "(590)"
    nm_table_2 = create_regra_orgao(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_mes_old_begin,
                                    date_mes_old_end, date_mes_current_begin, nm_intervalo, vistas_table=main_table)

    ## Regras Tutelas
    pacotes = "(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
    nm_tipo = 'tutela_investigacoes'
    cldc_dks = "(51219, 51220, 51221, 51222, 51223, 392, 395)"
    tppr_dks = "(-1)"
    nm_table_3 = create_regra_orgao(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_mes_old_begin,
                                    date_mes_old_end, date_mes_current_begin, nm_intervalo, vistas_table=main_table)

    table_orgao = spark.sql("""
        SELECT * FROM {0}
        UNION ALL
        SELECT * FROM {1}
        UNION ALL
        SELECT * FROM {2}
    """.format(nm_table_1, nm_table_2, nm_table_3))

    table_name = "{}.tb_detalhe_documentos_orgao".format(schema_exadata_aux)
    table_orgao.write.mode("overwrite").saveAsTable("temp_table_detalhe_documentos_orgao")
    temp_table = spark.table("temp_table_detalhe_documentos_orgao")
    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_detalhe_documentos_orgao")

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])
    spark.catalog.clearCache()


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
