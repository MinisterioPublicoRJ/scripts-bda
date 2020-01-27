import pyspark
from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp


spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_acervo_temporario") \
        .enableHiveSupport() \
        .getOrCreate()

table = spark.sql("""
        SELECT 
            docu_orgi_orga_dk_responsavel AS cod_orgao, 
            pacote_atribuicao,
            count(docu_dk) as acervo
        FROM exadata.mcpr_documento
            JOIN cluster.atualizacao_pj_pacote ON docu_orgi_orga_dk_responsavel = id_orgao
        WHERE 
            docu_fsdc_dk = 1
        GROUP BY docu_orgi_orga_dk_responsavel, pacote_atribuicao
""")

table.withColumn("data", from_unixtime(unix_timestamp(current_timestamp(), 'yyyy-MM-dd'), 'yyyy-MM-dd').cast('timestamp')) \
    .write.mode("overwrite").format("parquet").saveAsTable("exadata_aux.tb_acervo_diario")


