import sys, inspect
import pyspark
import argparse


def test_tb_acervo_todas_datas_presentes(spark, options):
    output = spark.sql("""
            select count(distinct dt_inclusao) as output
            from {0}.tb_acervo
            """.format(options['schema_exadata_aux']))\
        .collect()
    expected = spark.sql("""
            select datediff(max(dt_inclusao), min(dt_inclusao)) + 1 as expected_output
            from {0}.tb_acervo
            """.format(options['schema_exadata_aux']))\
        .collect()

    assert output == expected


def test_tb_saida_data_calculo_mais_recente(spark, options):
    output = spark.sql("""
            select distinct to_date(dt_calculo) as output from {0}.tb_saida
            """.format(options['schema_exadata_aux']))\
        .collect()
    expected = spark.sql("select to_date(current_timestamp()) as expected_output").collect()

    assert output == expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create table detalhe processo")
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    args = parser.parse_args()

    options = {
                'schema_exadata_aux': 'exadata_aux_dev'
              }

    spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("test_tabelas") \
        .enableHiveSupport() \
        .getOrCreate()

    functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    test_functions = [f for f in functions if f[0].startswith('test')]

    for test in test_functions:
        try:
            test[1](spark, options)
        except AssertionError:
            print("{}: FAILED".format(test[0]))
        except Exception as e:
            print("{}: ERROR - {}".format(test[0], e))
        else:
            print("{}: PASSED".format(test[0]))

