from datetime import datetime, timedelta, date

import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from utils import _update_impala_table
import argparse


# Nao e utilizada, porem fica como referencia para possivelmente o futuro
def create_regra_andamento(spark, options, nm_tipo, pacotes, tppr_dks, days_past=30):
    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    table = spark.sql("""
        SELECT orgao_id, '{6}' as tipo_detalhe, {3} as intervalo, orgi_nm_orgao, cod_pct,
            SUM(is_atual) as nr_andamentos_atual,
            SUM(CASE WHEN is_atual = 0 THEN 1 ELSE 0 END) as nr_andamentos_anterior,
            CASE
                WHEN (SUM(is_atual) - SUM(CASE WHEN is_atual = 0 THEN 1 ELSE 0 END)) = 0 THEN 0
                ELSE (SUM(is_atual) - SUM(CASE WHEN is_atual = 0 THEN 1 ELSE 0 END))/SUM(CASE WHEN is_atual = 0 THEN 1 ELSE 0 END)
            END as variacao_andamentos
        FROM (
            SELECT 
                CASE WHEN to_date(pcao_dt_andamento) >= to_date(date_sub(current_timestamp(), {3})) THEN 1 ELSE 0 END as is_atual,
                vist_orgi_orga_dk as orgao_id,
                orgi_nm_orgao,
                cod_pct
            FROM {0}.mcpr_documento A
            JOIN {0}.mcpr_vista B ON B.vist_docu_dk = A.DOCU_DK
            JOIN {0}.mcpr_andamento C ON C.pcao_vist_dk = B.vist_dk 
            JOIN {0}.mcpr_sub_andamento D ON D.stao_pcao_dk = C.pcao_dk
            JOIN {1}.atualizacao_pj_pacote p ON p.id_orgao = vist_orgi_orga_dk
            WHERE A.docu_tpst_dk != 11
            AND pcao_dt_cancelamento IS NULL
            AND to_date(pcao_dt_andamento) <= to_date(current_timestamp())
            AND to_date(pcao_dt_andamento) >= to_date(date_sub(current_timestamp(), {2}))
            AND stao_tppr_dk IN {5}
            AND cod_pct IN {4}
            ) t
        GROUP BY orgao_id, orgi_nm_orgao, cod_pct
    """.format(schema_exadata, schema_exadata_aux, 2*days_past, days_past, pacotes, tppr_dks, nm_tipo))
    nm_table = "ANDAMENTOS_{}_{}".format(nm_tipo, days_past)
    table.createOrReplaceTempView(nm_table)

    return nm_table


def create_regra_orgao(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_old_begin, date_old_end, date_current_begin, nm_intervalo):
    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    # LEFT JOIN ja que vistas podem n ter andamento associado ainda
    vistas = spark.sql("""
        SELECT 
            docu_dk,
            vist_dk, 
            CASE 
                WHEN docu_dt_cadastro >= '{2}' THEN 1 
                WHEN docu_dt_cadastro >= '{3}' AND docu_dt_cadastro <= '{7}' THEN 0
                ELSE NULL END AS is_instaurado_intervalo, 
            CASE WHEN vist_dt_abertura_vista >= '{2}' THEN 1 ELSE 0 END AS is_atual,
            vist_orgi_orga_dk,
            CASE WHEN stao_tppr_dk IN {6} AND pcao_dt_cancelamento IS NULL THEN 1 ELSE 0 END AS is_aproveitamento
        FROM {0}.mcpr_documento
        JOIN {0}.mcpr_vista ON docu_dk = vist_docu_dk
        LEFT JOIN {0}.mcpr_andamento ON pcao_vist_dk = vist_dk
        LEFT JOIN {0}.mcpr_sub_andamento ON stao_pcao_dk = pcao_dk
        JOIN {1}.atualizacao_pj_pacote ON id_orgao = vist_orgi_orga_dk
        WHERE vist_dt_abertura_vista >= '{3}'
        AND vist_dt_abertura_vista <= current_timestamp()
        AND (vist_dt_abertura_vista >= '{2}' OR vist_dt_abertura_vista <= '{7}')
        AND docu_cldc_dk IN {5}
        AND cod_pct IN {4}
        AND docu_tpst_dk != 11
    """.format(schema_exadata, schema_exadata_aux, date_current_begin, date_old_begin, pacotes, cldc_dks, tppr_dks, date_old_end))
    nm_table_vistas = "VISTAS_{}".format(nm_tipo)
    vistas.createOrReplaceTempView(nm_table_vistas)
    spark.catalog.cacheTable(nm_table_vistas)


    date_partition_current = datetime.now().date().strftime('%d%m%Y')
    date_partition_old = "".join(date_old_end.split('-')[::-1])

    # Dados de acervo de PIPs so comecaram a ser salvos a partir do dia 15-05-2020
    if '200' in pacotes and date_old_end < '2020-05-15':
        date_partition_old = "15052020"

    acervo = spark.sql("""
        SELECT
            tb_data_fim.cod_orgao as orgao_id,
            tb_data_fim.acervo_fim,
            tb_data_inicio.acervo_inicio,
            CASE WHEN (acervo_fim - acervo_inicio) = 0 THEN 0
                ELSE (acervo_fim - acervo_inicio)/acervo_inicio END as variacao
        FROM (
            SELECT cod_orgao, SUM(acervo) as acervo_inicio
            FROM {0}.tb_acervo
            JOIN {0}.atualizacao_pj_pacote ON id_orgao = cod_orgao
            WHERE dt_partition = '{3}'
            AND tipo_acervo IN {1}
            AND cod_pct IN {2}
            GROUP BY cod_orgao
            ) tb_data_inicio
        RIGHT JOIN (
            SELECT cod_orgao, SUM(acervo) as acervo_fim
            FROM {0}.tb_acervo
            JOIN {0}.atualizacao_pj_pacote ON id_orgao = cod_orgao
            WHERE dt_partition = '{4}'
            AND tipo_acervo IN {1}
            AND cod_pct IN {2}
            GROUP BY cod_orgao
            ) tb_data_fim ON tb_data_fim.cod_orgao = tb_data_inicio.cod_orgao
    """.format(schema_exadata_aux, cldc_dks, pacotes, date_partition_old, date_partition_current))
    nm_table_acervo = "ACERVO_{}".format(nm_tipo)
    acervo.createOrReplaceTempView(nm_table_acervo)

    # SUM de is_instaurado, ja que os instaurados atuais sao sempre 1
    atuais = spark.sql("""
        SELECT
            vist_orgi_orga_dk,
            COUNT(DISTINCT docu_dk) as nr_documentos_distintos_atual,
            SUM(nr_aberturas_vista) as nr_aberturas_vista_atual,
            SUM(has_aproveitamento) as nr_aproveitamentos_atual,
            nvl(SUM(is_instaurado), 0) as nr_instaurados_atual
        FROM (
            SELECT 
                vist_orgi_orga_dk,
                docu_dk,
                COUNT(vist_dk) AS nr_aberturas_vista,
                MAX(is_aproveitamento) AS has_aproveitamento,
                MAX(is_instaurado_intervalo) AS is_instaurado
            FROM {0} t
            WHERE is_atual = 1
            GROUP BY vist_orgi_orga_dk, docu_dk) t
        GROUP BY vist_orgi_orga_dk
    """.format(nm_table_vistas))
    nm_table_atuais = "ATUAIS_{}".format(nm_tipo)
    atuais.createOrReplaceTempView(nm_table_atuais)

    # Count de is_instaurado, pois quando is_atual = 0, n ha is_instaurado = 1
    anteriores = spark.sql("""
        SELECT
            vist_orgi_orga_dk,
            COUNT(DISTINCT docu_dk) as nr_documentos_distintos_anterior,
            SUM(nr_aberturas_vista) as nr_aberturas_vista_anterior,
            SUM(has_aproveitamento) as nr_aproveitamentos_anterior,
            nvl(COUNT(is_instaurado), 0) AS nr_instaurados_anterior
        FROM (
            SELECT 
                vist_orgi_orga_dk,
                docu_dk,
                COUNT(vist_dk) AS nr_aberturas_vista,
                MAX(is_aproveitamento) AS has_aproveitamento,
                MAX(is_instaurado_intervalo) AS is_instaurado
            FROM {0} t
            WHERE is_atual = 0
            GROUP BY vist_orgi_orga_dk, docu_dk) t
        GROUP BY vist_orgi_orga_dk
    """.format(nm_table_vistas))
    nm_table_anteriores = "ANTERIORES_{}".format(nm_tipo)
    anteriores.createOrReplaceTempView(nm_table_anteriores)

    table = spark.sql("""
        SELECT '{0}' as tipo_detalhe, '{1}' as intervalo, orgi_nm_orgao, cod_pct,
        at.*, ac.acervo_inicio, ac.acervo_fim, ac.variacao as variacao_acervo,
        an.nr_documentos_distintos_anterior, an.nr_aberturas_vista_anterior, an.nr_aproveitamentos_anterior, an.nr_instaurados_anterior,
        CASE WHEN (at.nr_documentos_distintos_atual - an.nr_documentos_distintos_anterior) = 0 THEN 0
            ELSE (at.nr_documentos_distintos_atual - an.nr_documentos_distintos_anterior)/an.nr_documentos_distintos_anterior END as variacao_documentos_distintos,
        CASE WHEN (at.nr_aberturas_vista_atual - an.nr_aberturas_vista_anterior) = 0 THEN 0
            ELSE (at.nr_aberturas_vista_atual - an.nr_aberturas_vista_anterior)/an.nr_aberturas_vista_anterior END as variacao_aberturas_vista,
        CASE WHEN (at.nr_aproveitamentos_atual - an.nr_aproveitamentos_anterior) = 0 THEN 0
            ELSE (at.nr_aproveitamentos_atual - an.nr_aproveitamentos_anterior)/an.nr_aproveitamentos_anterior END as variacao_aproveitamentos,
        CASE WHEN (at.nr_instaurados_atual - an.nr_instaurados_anterior) = 0 THEN 0
            ELSE (at.nr_instaurados_atual - an.nr_instaurados_anterior)/an.nr_instaurados_anterior END as variacao_instaurados
        FROM {2} at
        JOIN {5}.atualizacao_pj_pacote ON id_orgao = vist_orgi_orga_dk
        LEFT JOIN {3} an ON at.vist_orgi_orga_dk = an.vist_orgi_orga_dk
        LEFT JOIN {4} ac ON ac.orgao_id = at.vist_orgi_orga_dk
    """.format(nm_tipo, nm_intervalo, nm_table_atuais, nm_table_anteriores, nm_table_acervo, schema_exadata_aux))
    nm_table_final = "DETALHE_ORGAO_{}_{}".format(nm_tipo, nm_intervalo)
    table.createOrReplaceTempView(nm_table_final)

    return nm_table_final


def create_regra_cpf(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_old_begin, date_old_end, date_current_begin, nm_intervalo):
    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    # LEFT JOIN ja que vistas podem n ter andamento associado ainda
    vistas = spark.sql("""
        SELECT 
            docu_dk,
            vist_dk, 
            CASE 
                WHEN docu_dt_cadastro >= '{2}' THEN 1 
                WHEN docu_dt_cadastro >= '{3}' AND docu_dt_cadastro <= '{7}' THEN 0
                ELSE NULL END AS is_instaurado_intervalo, 
            CASE WHEN vist_dt_abertura_vista >= '{2}' THEN 1 ELSE 0 END AS is_atual,
            vist_orgi_orga_dk,
            pesf_cpf,
            CASE WHEN stao_tppr_dk IN {6} AND pcao_dt_cancelamento IS NULL THEN 1 ELSE 0 END AS is_aproveitamento
        FROM {0}.mcpr_documento
        JOIN {0}.mcpr_vista ON docu_dk = vist_docu_dk
        LEFT JOIN {0}.mcpr_andamento ON pcao_vist_dk = vist_dk
        LEFT JOIN {0}.mcpr_sub_andamento ON stao_pcao_dk = pcao_dk
        JOIN {1}.atualizacao_pj_pacote ON id_orgao = vist_orgi_orga_dk
        JOIN {0}.mcpr_pessoa_fisica ON vist_pesf_pess_dk_resp_andam = pesf_pess_dk
        WHERE vist_dt_abertura_vista >= '{3}'
        AND vist_dt_abertura_vista <= current_timestamp()
        AND (vist_dt_abertura_vista >= '{2}' OR vist_dt_abertura_vista <= '{7}')
        AND docu_cldc_dk IN {5}
        AND cod_pct IN {4}
        AND docu_tpst_dk != 11
    """.format(schema_exadata, schema_exadata_aux, date_current_begin, date_old_begin, pacotes, cldc_dks, tppr_dks, date_old_end))
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
            nvl(SUM(is_instaurado), 0) as nr_instaurados_atual
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
            nvl(COUNT(is_instaurado), 0) as nr_instaurados_anterior
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
        SELECT '{0}' as tipo_detalhe, '{1}' as intervalo, 
        at.*, 
        an.nr_documentos_distintos_anterior,
        an.nr_aberturas_vista_anterior,
        an.nr_aproveitamentos_anterior,
        an.nr_instaurados_anterior,
        CASE WHEN (at.nr_documentos_distintos_atual - an.nr_documentos_distintos_anterior) = 0 THEN 0
            ELSE (at.nr_documentos_distintos_atual - an.nr_documentos_distintos_anterior)/an.nr_documentos_distintos_anterior END as variacao_documentos_distintos,
        CASE WHEN (at.nr_aberturas_vista_atual - an.nr_aberturas_vista_anterior) = 0 THEN 0
            ELSE (at.nr_aberturas_vista_atual - an.nr_aberturas_vista_anterior)/an.nr_aberturas_vista_anterior END as variacao_aberturas_vista,
        CASE WHEN (at.nr_aproveitamentos_atual - an.nr_aproveitamentos_anterior) = 0 THEN 0
            ELSE (at.nr_aproveitamentos_atual - an.nr_aproveitamentos_anterior)/an.nr_aproveitamentos_anterior END as variacao_aproveitamentos,
        CASE WHEN (at.nr_instaurados_atual - an.nr_instaurados_anterior) = 0 THEN 0
            ELSE (at.nr_instaurados_atual - an.nr_instaurados_anterior)/an.nr_instaurados_anterior END as variacao_instaurados
        FROM {2} at
        LEFT JOIN {3} an ON at.vist_orgi_orga_dk = an.vist_orgi_orga_dk
            AND at.pesf_cpf = an.pesf_cpf
    """.format(nm_tipo, nm_intervalo, nm_table_atuais, nm_table_anteriores))
    nm_table_final = "DETALHE_CPF_{}_{}".format(nm_tipo, nm_intervalo)
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

    # Calcula datas para o mes corrente e mes anterior ate a mesma data (ou mais proxima)
    date_today = date.today()

    date_mes_current_begin = date_today.strftime('%Y-%m-01')

    mid_last_month = (date_today.replace(day=15) - timedelta(30))
    date_mes_old_begin = mid_last_month.strftime('%Y-%m-01')

    day = date_today.day
    if day <= 28:
        date_mes_old_end = mid_last_month.replace(day=day).strftime('%Y-%m-%d')
    else:
        while True:
            try:
                date_mes_old_end = mid_last_month.replace(day=day).strftime('%Y-%m-%d')
                break
            except:
                day -= 1

    # Tabela agregada orgao cpf
    ## Regras PIPs
    pacotes = "(200)"
    nm_intervalo = 'mes'
    nm_tipo = 'pip_inqueritos'
    cldc_dks = "(3, 494)"
    tppr_dks = ("(6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,"
                "6346,6350,6359,6392,6017,6018,6020,7745,6648,6649,6650,6651,6652,6653,6654,"
                "6038,6039,6040,6041,6042,6043,7815,7816,6620,6257,6258,7878,7877,6367,6368,6369,6370,1208,1030,6252,6253,1201,1202,6254)")
    nm_table_1 = create_regra_cpf(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_mes_old_begin, date_mes_old_end, date_mes_current_begin, nm_intervalo)

    nm_tipo = 'pip_pics'
    cldc_dks = "(590)"
    nm_table_2 = create_regra_cpf(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_mes_old_begin, date_mes_old_end, date_mes_current_begin, nm_intervalo)

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
    spark.catalog.clearCache()


    # Tabela agregada orgao
    ## Regras PIPs
    pacotes = "(200)"
    nm_intervalo = 'mes'

    nm_tipo = 'pip_inqueritos'
    cldc_dks = "(3, 494)"
    tppr_dks = ("(6549,6593,6591,6343,6338,6339,6340,6341,6342,7871,7897,7912,"
                "6346,6350,6359,6392,6017,6018,6020,7745,6648,6649,6650,6651,6652,6653,6654,"
                "6038,6039,6040,6041,6042,6043,7815,7816,6620,6257,6258,7878,7877,6367,6368,6369,6370,1208,1030,6252,6253,1201,1202,6254)")
    nm_table_1 = create_regra_orgao(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_mes_old_begin, date_mes_old_end, date_mes_current_begin, nm_intervalo)

    nm_tipo = 'pip_pics'
    cldc_dks = "(590)"
    nm_table_2 = create_regra_orgao(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_mes_old_begin, date_mes_old_end, date_mes_current_begin, nm_intervalo)

    ## Regras Tutelas
    pacotes = "(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"

    nm_tipo = 'tutela_investigacoes'
    cldc_dks = "(51219, 51220, 51221, 51222, 51223, 392, 395)"
    tppr_dks = "(-1)"
    nm_table_3 = create_regra_orgao(spark, options, nm_tipo, pacotes, cldc_dks, tppr_dks, date_mes_old_begin, date_mes_old_end, date_mes_current_begin, nm_intervalo)

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

    # Tabela andamentos orgao
    # nm_tipo = 'tutela_processos'
    # days_past = 30
    # pacotes = "(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
    # tppr_dks = "(6251)"
    # nm_table_1 = create_regra_andamento(spark, options, nm_tipo, pacotes, tppr_dks, days_past)

    # days_past = 60
    # nm_table_2 = create_regra_andamento(spark, options, nm_tipo, pacotes, tppr_dks, days_past)

    # days_past = 365
    # nm_table_3 = create_regra_andamento(spark, options, nm_tipo, pacotes, tppr_dks, days_past)

    # table_orgao = spark.sql("""
    #     SELECT * FROM {0}
    #     UNION ALL
    #     SELECT * FROM {1}
    #     UNION ALL
    #     SELECT * FROM {2}
    # """.format(nm_table_1, nm_table_2, nm_table_3))

    # table_name = "{}.tb_detalhe_andamentos_orgao".format(schema_exadata_aux)
    # table_orgao.write.mode("overwrite").saveAsTable("temp_table_detalhe_andamentos_orgao")
    # temp_table = spark.table("temp_table_detalhe_andamentos_orgao")
    # temp_table.write.mode("overwrite").saveAsTable(table_name)
    # spark.sql("drop table temp_table_detalhe_andamentos_orgao")

    # _update_impala_table(table_name, options['impala_host'], options['impala_port'])
    # spark.catalog.clearCache()



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
