import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from utils import _update_impala_table
from decouple import config


def check_table_exists(schema, table_name):
    spark.sql("use %s" % schema)
    result_table_check = spark.sql("SHOW TABLES LIKE '%s'" % table_name).count()
    return True if result_table_check > 0 else False

spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_acervo") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .enableHiveSupport() \
        .getOrCreate()

schema_exadata = config('SCHEMA_EXADATA')
schema_exadata_aux = config('SCHEMA_EXADATA_AUX')

table = spark.sql("""
        SELECT 
            docu_orgi_orga_dk_responsavel AS cod_orgao, 
            cod_pct as cod_atribuicao,
            count(docu_dk) as acervo,
            docu_cldc_dk as tipo_acervo
        FROM {}.mcpr_documento
            JOIN {}.atualizacao_pj_pacote ON docu_orgi_orga_dk_responsavel = id_orgao
            
        WHERE 
            docu_fsdc_dk = 1
        GROUP BY docu_orgi_orga_dk_responsavel, cod_pct, docu_cldc_dk
""".format(schema_exadata, schema_exadata_aux))

table = table.withColumn(
        "dt_inclusao",
        from_unixtime(
            unix_timestamp(current_timestamp(), 'yyyy-MM-dd'), 'yyyy-MM-dd') \
        .cast('timestamp')) \
        .withColumn("dt_partition", date_format(current_timestamp(), "ddMMyyyy"))


is_exists_table_acervo = check_table_exists(schema_exadata_aux, "tb_acervo")

table_name = "{}.tb_acervo".format(schema_exadata_aux)

if is_exists_table_acervo:
    table.coalesce(1).write.mode("overwrite").insertInto(table_name, overwrite=True)
else:
    table.write.partitionBy("dt_partition").mode("overwrite").saveAsTable(table_name)

_update_impala_table(table_name)