from datetime import datetime, timedelta

import pyspark
import argparse

from utils import _update_impala_table


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_indice_sucesso") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    days_past_start = options['days_past_start']
    days_past_end = options['days_past_end']

    grupo = spark.sql(
        """
        SELECT
            pip_codigo as orgao,
            count(Distinct vist_docu_dk) as vistas
        FROM {0}.mcpr_documento
            JOIN {0}.mcpr_vista ON vist_docu_dk = docu_dk
            JOIN (
                SELECT pip_codigo_antigo, pip_codigo 
                from {1}.temp_pip_aisp 
                GROUP BY pip_codigo_antigo, pip_codigo
            ) p ON p.pip_codigo_antigo = vist_orgi_orga_dk
            -- este operador OR pode duplicar as linhas?
            OR p.pip_codigo = vist_orgi_orga_dk
            JOIN {0}.mcpr_pessoa_fisica pess ON pess.pesf_pess_dk = vist_pesf_pess_dk_resp_andam
            JOIN {0}.rh_funcionario f ON pess.pesf_cpf = f.cpf
        WHERE docu_cldc_dk IN (3, 494, 590) -- PIC e Inqueritos
            AND vist_dt_abertura_vista >= cast(date_sub(current_timestamp(), {2}) as timestamp)
            AND vist_dt_abertura_vista <= cast(date_sub(current_timestamp(), {3}) as timestamp)
            AND f.cdtipfunc IN ('1', '2')
        GROUP BY pip_codigo
        """.format(schema_exadata, schema_exadata_aux, days_past_start, days_past_end)
    )
    grupo.createOrReplaceTempView('grupo')

    denuncia = spark.sql(
        """
        SELECT
            pip_codigo as orgao,
            count(Distinct vist_docu_dk) as denuncias
        FROM {0}.mcpr_documento
            JOIN {0}.mcpr_vista ON vist_docu_dk = docu_dk
            JOIN {0}.mcpr_andamento ON pcao_vist_dk = vist_dk
            JOIN {0}.mcpr_sub_andamento ON stao_pcao_dk = pcao_dk
            JOIN (
                SELECT pip_codigo_antigo, pip_codigo
                from {1}.temp_pip_aisp 
                GROUP BY pip_codigo_antigo, pip_codigo
            ) p ON p.pip_codigo_antigo = vist_orgi_orga_dk
            JOIN {0}.mcpr_pessoa_fisica pess ON pess.pesf_pess_dk = vist_pesf_pess_dk_resp_andam
            JOIN {0}.rh_funcionario f ON pess.pesf_cpf = f.cpf
        WHERE docu_cldc_dk IN (3, 494, 590) -- PIC e Inqueritos
            AND vist_dt_abertura_vista >= cast(date_sub(current_timestamp(), {2}) as timestamp)
            AND f.cdtipfunc IN ('1', '2')
            AND stao_tppr_dk IN (6252, 6253, 1201, 1202, 6254)
        GROUP BY pip_codigo
        """.format(schema_exadata, schema_exadata_aux, days_past_start)
    )
    denuncia.createOrReplaceTempView('denuncia')

    indice_elucidacao = spark.sql(
        """
            SELECT 
                g.orgao,
                g.vistas,
                d.denuncias,
                (d.denuncias/g.vistas) AS rate
            FROM grupo g
            JOIN denuncia d ON g.orgao = d.orgao
        """.format(schema_exadata, schema_exadata_aux)
    )

    table_name = "{}.tb_indicadores_sucesso".format(schema_exadata_aux)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table distribuicao entradas")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    parser.add_argument('-c','--daysPastStart', metavar='daysPastStart', type=int, default=540, help='')
    parser.add_argument('-f','--daysPastEnd', metavar='daysPastEnd', type=int, default=180, help='')
    
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort,
                    'days_past_start' : args.daysPastStart,
                    'days_past_end': args.daysPastEnd
                }

    execute_process(options)
