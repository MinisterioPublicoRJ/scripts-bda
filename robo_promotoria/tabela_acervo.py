import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from utils import _update_impala_table


def check_table_exists(schema, table_name):
    spark.sql("use %s" % schema)
    result_table_check = spark.sql("SHOW TABLES LIKE '%s'" % table_name).count()
    return True if result_table_check > 0 else False

spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_acervo") \
        .enableHiveSupport() \
        .getOrCreate()

spark.conf.set("spark.sql.sources.partitionOverwriteMode","dynamic")


table = spark.sql("""
        SELECT 
            docu_orgi_orga_dk_responsavel AS cod_orgao, 
            pacote_atribuicao,
            count(docu_dk) as acervo
        FROM exadata_dev.mcpr_documento
            JOIN cluster.atualizacao_pj_pacote ON docu_orgi_orga_dk_responsavel = id_orgao
        WHERE 
            docu_fsdc_dk = 1
        GROUP BY docu_orgi_orga_dk_responsavel, pacote_atribuicao
""")

table = table.withColumn("tipo_acervo", lit(0)).withColumn(
        "dt_inclusao",
        from_unixtime(
            unix_timestamp(current_timestamp(), 'yyyy-MM-dd'), 'yyyy-MM-dd') \
        .cast('timestamp')) \
        .withColumn("dt_partition", date_format(current_timestamp(), "ddMMyyyy"))


is_exists_table_acervo = check_table_exists("exadata_aux", "tb_acervo")

if is_exists_table_acervo:
    table.coalesce(1).write.mode("overwrite").insertInto("exadata_aux.tb_acervo", overwrite=True)
else:
    table.write.partitionBy("dt_partition").mode("overwrite").saveAsTable("exadata_aux.tb_acervo")

_update_impala_table("exadata_aux.tb_acervo")