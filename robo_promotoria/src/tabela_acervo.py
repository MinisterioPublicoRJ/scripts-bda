import argparse
import pyspark

from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from generic_utils import execute_compute_stats

def check_table_exists(spark, schema, table_name):
    spark.sql("use %s" % schema)
    result_table_check = spark.sql("SHOW TABLES LIKE '%s'" % table_name).count()
    return True if result_table_check > 0 else False


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_acervo") \
            .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']
    table_name = options['table_name']

    table = spark.sql("""
            SELECT 
                docu_orgi_orga_dk_responsavel AS cod_orgao, 
                cod_pct as cod_atribuicao,
                count(docu_dk) as acervo,
                docu_cldc_dk as tipo_acervo
            FROM {}.mcpr_documento
                LEFT JOIN {}.atualizacao_pj_pacote ON docu_orgi_orga_dk_responsavel = id_orgao
                
            WHERE 
                docu_fsdc_dk = 1
            AND docu_tpst_dk != 11
            GROUP BY docu_orgi_orga_dk_responsavel, cod_pct, docu_cldc_dk
    """.format(schema_exadata, schema_exadata_aux))

    table = table.withColumn(
            "dt_inclusao",
            from_unixtime(
                unix_timestamp(current_timestamp(), 'yyyy-MM-dd'), 'yyyy-MM-dd') \
            .cast('timestamp')) \
            .withColumn("dt_partition", date_format(current_timestamp(), "ddMMyyyy"))


    is_exists_table_acervo = check_table_exists(spark, schema_exadata_aux, table_name)

    table_name = "{}.{}".format(schema_exadata_aux, table_name)

    if is_exists_table_acervo:
        table.coalesce(1).write.mode("overwrite").insertInto(table_name, overwrite=True)
    else:
        table.write.partitionBy("dt_partition").mode("overwrite").saveAsTable(table_name)

    execute_compute_stats(table_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table acervo")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    parser.add_argument('-t','--tableName', metavar='tableName', type=str, help='')
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort,
                    'table_name' : args.tableName,
                }

    execute_process(options)