import argparse
import pyspark

from datetime import datetime
from pyspark.sql.functions import (
    unix_timestamp,
    from_unixtime,
    current_timestamp,
    date_format
)
from generic_utils import execute_compute_stats

def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_distribuicao") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata_aux = options['schema_exadata_aux']
    table_name = options['table_name']

    date_now = datetime.now()
    data_atual = date_now.strftime("%Y-%m-%d")

    qtd_acervo = spark.sql(
        """
        select A.cod_orgao, A.cod_atribuicao as cod_atribuicao, SUM(A.acervo) as acervo
        from {0}.tb_acervo A
        inner join {0}.tb_regra_negocio_investigacao B
        on A.cod_atribuicao = B.cod_atribuicao AND A.tipo_acervo = B.classe_documento
        where A.dt_inclusao = '{1}'
        group by A.cod_orgao, A.cod_atribuicao
        """.format(schema_exadata_aux, data_atual)
    )
    qtd_acervo.registerTempTable('qtd_acervo_table')

    estatisticas = spark.sql(
        """
        select cod_orgao, acervo, dist.*
        from qtd_acervo_table
        inner join (
            select cod_atribuicao,
            min(acervo) as minimo,
            max(acervo) as maximo,
            avg(acervo) as media,
            percentile(acervo, 0.25) as primeiro_quartil,
            percentile(acervo, 0.5) as mediana,
            percentile(acervo, 0.75) as terceiro_quartil,
            percentile(acervo, 0.75) - percentile(acervo, 0.25) as IQR,
            percentile(acervo, 0.25)
                - 1.5*(percentile(acervo, 0.75) - percentile(acervo, 0.25)) as Lout,
            percentile(acervo, 0.75)
                + 1.5*(percentile(acervo, 0.75) - percentile(acervo, 0.25)) as Hout
            from qtd_acervo_table t 
            group by cod_atribuicao) dist ON dist.cod_atribuicao = qtd_acervo_table.cod_atribuicao
        """
    ).withColumn(
        "dt_inclusao",
        from_unixtime(
            unix_timestamp(current_timestamp(), 'yyyy-MM-dd HH:mm:ss'), 'yyyy-MM-dd HH:mm:ss')
        .cast('timestamp')
    )

    table_name = "{}.{}".format(schema_exadata_aux, table_name)

    estatisticas.write.mode("overwrite").saveAsTable("temp_table_distribuicao")
    temp_table = spark.table("temp_table_distribuicao")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_distribuicao")

    execute_compute_stats(table_name)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table distribuicao")
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    parser.add_argument('-t','--tableName', metavar='tableName', type=str, help='')
    args = parser.parse_args()

    options = {
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort,
                    'table_name' : args.tableName,
                }

    execute_process(options)