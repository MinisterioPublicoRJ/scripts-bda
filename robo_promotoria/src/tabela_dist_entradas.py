import pyspark
from datetime import datetime, timedelta
import argparse

from pyspark.sql.functions import (
    unix_timestamp,
    from_unixtime,
    current_timestamp,
    date_format
)

from utils import _update_impala_table


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_distribuicao_entradas") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    nb_past_days = options['past_days']

    date_range = spark.createDataFrame(
        [
            {'dt': (datetime.now() - timedelta(i)).strftime('%Y-%m-%d')}
            for i in range(nb_past_days)
        ]
    )
    date_range.registerTempTable("date_range")

    comb_dates = spark.sql(
        """
        SELECT date_range.dt,
        vist_orgi_orga_dk as comb_orga_dk,
        pesf_cpf as comb_cpf
        FROM date_range CROSS JOIN (
            SELECT DISTINCT vist_orgi_orga_dk, pesf_cpf 
            FROM {0}.mcpr_vista v
            JOIN {0}.mcpr_pessoa_fisica p ON v.VIST_PESF_PESS_DK_RESP_ANDAM = p.PESF_PESS_DK
            JOIN {0}.rh_funcionario rf ON p.pesf_cpf = rf.CPF
            WHERE to_date(VIST_DT_ABERTURA_VISTA) >= date_sub(to_date(current_timestamp()), {1})
            AND p.pesf_aplicacao_atualizou = 'RH' AND rf.cdtipfunc = '1'
        ) t2
        WHERE dayofweek(date_range.dt) NOT IN (1, 7)
        """.format(schema_exadata, nb_past_days)
    )
    comb_dates.registerTempTable('date_combs')

    entradas_table = spark.sql(
        """
        SELECT dt,
        comb_orga_dk,
        comb_cpf,
        nvl(COUNT(vist_dt_abertura_vista), 0) as nr_entradas
        FROM {0}.mcpr_vista v
        JOIN {0}.mcpr_documento ON docu_dk = vist_docu_dk
        JOIN {0}.mcpr_pessoa_fisica p ON v.VIST_PESF_PESS_DK_RESP_ANDAM = p.PESF_PESS_DK
        JOIN {0}.rh_funcionario rf ON p.pesf_cpf = rf.CPF
        RIGHT JOIN date_combs c 
            ON comb_orga_dk = vist_orgi_orga_dk
            AND comb_cpf = p.pesf_cpf
            AND dt = to_date(v.vist_dt_abertura_vista)
        WHERE (DOCU_TPST_DK != 11 OR DOCU_TPST_DK IS NULL)
        GROUP BY dt, comb_orga_dk, comb_cpf
        """.format(schema_exadata)
    )
    entradas_table.registerTempTable('entradas_table')

    estatisticas = spark.sql(
        """
        SELECT 
            t2.nr_entradas as nr_entradas_hoje, 
            t1.* 
        FROM (
            SELECT
                comb_orga_dk,
                comb_cpf,
                min(nr_entradas) as minimo,
                max(nr_entradas) as maximo,
                avg(nr_entradas) as media,
                percentile(nr_entradas, 0.25) as primeiro_quartil,
                percentile(nr_entradas, 0.5) as mediana,
                percentile(nr_entradas, 0.75) as terceiro_quartil,
                percentile(nr_entradas, 0.75) - percentile(nr_entradas, 0.25) as IQR,
                percentile(nr_entradas, 0.25) - 1.5*(percentile(nr_entradas, 0.75)
                    - percentile(nr_entradas, 0.25)) as Lout,
                percentile(nr_entradas, 0.75) + 1.5*(percentile(nr_entradas, 0.75)
                    - percentile(nr_entradas, 0.25)) as Hout
            FROM entradas_table t
            GROUP BY comb_orga_dk, comb_cpf
        ) t1
        LEFT JOIN entradas_table t2 ON t2.comb_orga_dk = t1.comb_orga_dk
        AND t2.comb_cpf = t1.comb_cpf
        AND t2.dt = to_date(current_timestamp())
        """
    )


    table_name = "{}.tb_dist_entradas".format(schema_exadata_aux)

    estatisticas.write.mode("overwrite").saveAsTable("temp_table_dist_entrada")
    temp_table = spark.table("temp_table_dist_entrada")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_dist_entrada")

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table distribuicao entradas")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    parser.add_argument('-p','--pastDays', metavar='pastDays', type=int, default=60, help='')
    
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort,
                    'past_days' : args.pastDays
                }

    execute_process(options)
