import pyspark
from datetime import datetime
import argparse

from pyspark.sql.functions import (
    unix_timestamp,
    from_unixtime,
    current_timestamp,
    date_format
)

from utils import _update_impala_table


def check_table_exists(spark, schema, table_name):
    spark.sql("use %s" % schema)
    result_table_check = spark.sql("SHOW TABLES LIKE '%s'" % table_name).count()
    return True if result_table_check > 0 else False


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_distribuicao") \
            .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata_aux = options['schema_exadata_aux']

    date_now = datetime.now()
    data_atual = date_now.strftime("%Y-%m-%d")

    estatisticas = spark.sql(
        """
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
        from (
            select A.cod_orgao, A.cod_atribuicao as cod_atribuicao, SUM(A.acervo) as acervo
            from {0}.tb_acervo A
            inner join {0}.tb_regra_negocio_investigacao B
            on A.cod_atribuicao = B.cod_atribuicao AND A.tipo_acervo = B.classe_documento
            where A.dt_inclusao = '{1}'
            group by A.cod_orgao, A.cod_atribuicao
        ) t 
        group by cod_atribuicao
        """.format(schema_exadata_aux, data_atual)
    ).withColumn(
        "dt_inclusao",
        from_unixtime(
            unix_timestamp(current_timestamp(), 'yyyy-MM-dd'), 'yyyy-MM-dd')
        .cast('timestamp')
    ).withColumn("dt_partition", date_format(current_timestamp(), "ddMMyyyy"))


    is_exists_table_distribuicao = check_table_exists(spark, schema_exadata_aux, "tb_distribuicao")

    table_name = "{}.tb_distribuicao".format(schema_exadata_aux)

    if is_exists_table_distribuicao:
        estatisticas.coalesce(1).write.mode("overwrite").insertInto(table_name, overwrite=True)
    else:
        estatisticas.write.partitionBy("dt_partition").saveAsTable(table_name)

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table distribuicao")
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    args = parser.parse_args()

    options = {
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort
                }

    execute_process(options)