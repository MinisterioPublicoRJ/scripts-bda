import pyspark

import argparse

from tramitacao.tutela_acoes import execute_process as acoes_process_2
from tramitacao.utils_tempo import execute_process as processa_regra
from generic_utils import execute_compute_stats


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
    parser.add_argument(
        '-t',
        '--tableName',
        metavar='tableName',
        type=str,
        help=''
    )
    args = parser.parse_args()

    options = {
        'schema_exadata': args.schemaExadata,
        'schema_exadata_aux': args.schemaExadataAux,
        'impala_host': args.impalaHost,
        'impala_port': args.impalaPort,
        'table_name': args.tableName,
    }

    # Regras
    classes_1 = "(392)"
    andamentos_1 = """(7912,6548,6326,6681,6678,6645,6682,6680,
                    6679,6644,6668,6666,6665,6669,6667,6664,
                    6655,6662,6659,6658,6663,6661,6660,6657,
                    6670,6676,6674,6673,6677,6675,6672,6018,
                    6341,6338,6019,6017,6591,6339,6553,7871,
                    6343,6340,6342,6021,6334,6331,6022,6020,
                    6593,6332,7872,6336,6333,6335,7745,6346,
                    6345,6015,6016,6325,6327,6328,6329,6330,
                    6337,6344,6656,6671,7869,7870,6324,6251)"""
    pacotes_1 = "(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
    nm_table_1 = processa_regra(spark, options, classes_1, andamentos_1, pacotes_1, 'tutela_inqueritos_civis', 5)
    
    classes_2 = "(18, 126, 127, 159, 175, 176, 177, 441)"
    andamentos_2 = "(6374,6375,6376,6377,6378)"
    pacotes_2 = "(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)"
    nm_table_2 = processa_regra(spark, options, classes_2, andamentos_2, pacotes_2, 'tutela_acoes_sentenca', 5)

    acoes_process_2(spark, options)

    classes_3 = "(3, 494, 590)"
    andamentos_3 = """(6682,6669,6018,6341,6338,6019,6017,6591,6339,7871,
                       6343,6340,6342,7745,6346,7915,6272,6253,6392,6377,
                       6378,6359,6362,6361,6436,6524,7737,7811,6625,6718)"""
    pacotes_3 = "(200, 201, 202, 203, 204, 205, 206, 207, 208, 209)"
    nm_table_3 = processa_regra(spark, options, classes_3, andamentos_3, pacotes_3, 'pip_investigacoes', 1)

    tramitacao_final = spark.sql("""
        select * from {0}
        UNION ALL
        select * from {1}
        UNION ALL
        select * from {2}
        UNION ALL
        select * from {3}
        """.format(nm_table_1, nm_table_2, "tutela_final_acoes_tempo_2", nm_table_3)
    )

    table_name = options['table_name']
    table_name = "{}.{}".format(
        options["schema_exadata_aux"], table_name
    )

    tramitacao_final.write.mode("overwrite").saveAsTable("temp_table_tempo_tramitacao_integrado")
    temp_table = spark.table("temp_table_tempo_tramitacao_integrado")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_tempo_tramitacao_integrado")

    execute_compute_stats(table_name)
