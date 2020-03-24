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
        select
         ic.id_orgao,
          media_orgao, minimo_orgao,maximo_orgao,mediana_orgao,
          media_pacote, minimo_pacote, maximo_pacote, mediana_pacote,
          media_pacote_t1, minimo_pacote_t1, maximo_pacote_t1, mediana_pacote_t1,
          media_orgao_t1, minimo_orgao_t1, maximo_orgao_t1, mediana_orgao_t1,
          media_pacote_t2, minimo_pacote_t2, maximo_pacote_t2, mediana_pacote_t2,
          media_orgao_t2, minimo_orgao_t2, maximo_orgao_t2, mediana_orgao_t2
       from tramitacao_ic_final ic
        join tramitacao_acoes_final acoes on ic.id_orgao = acoes.id_orgao
        """
    )

    table_name = "{}.tb_tempo_tramitacao".format(
        options["schema_exadata_aux"]
    )

    tramitacao_final.write.mode("overwrite").saveAsTable("temp_table_tempo_tramitacao")
    temp_table = spark.table("temp_table_tempo_tramitacao")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_tempo_tramitacao")

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])
