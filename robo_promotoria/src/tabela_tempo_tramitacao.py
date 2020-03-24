import pyspark

from utils import _update_impala_table
import argparse

from tramitacao.inquerito_civil import execute_process as inquerito_process
from tramitacao.acoes import execute_process as acoes_process


if __name__ == "__main__":
    spark = pyspark.sql.session.SparkSession \
                    .builder \
                    .appName("Processamento Tempo Tramitaca") \
                    .enableHiveSupport() \
                    .getOrCreate()

    parser = argparse.ArgumentParser(
        description="Create table radar performance"
    )
    parser.add_argument(
        '-e',
        '--schemaExadata',
        metavar='schemaExadata',
        type=str,
        help=''
    )
    parser.add_argument(
        '-a',
        '--schemaExadataAux',
        metavar='schemaExadataAux',
        type=str,
        help=''
    )
    parser.add_argument(
        '-i',
        '--impalaHost',
        metavar='impalaHost',
        type=str,
        help=''
    )
    parser.add_argument(
        '-o',
        '--impalaPort',
        metavar='impalaPort',
        type=str,
        help=''
    )
    args = parser.parse_args()

    options = {
        'schema_exadata': args.schemaExadata,
        'schema_exadata_aux': args.schemaExadataAux,
        'impala_host': args.impalaHost,
        'impala_port': args.impalaPort,
    }

    inquerito_process(spark, options)
    acoes_process(spark, options)
    tramitacao_final = spark.sql("""
        select * from tramitacao_ic_final ic
        join tramitacao_acoes_final acoes on ic.id_orgao = acoes.id_orgao
        """
    )
