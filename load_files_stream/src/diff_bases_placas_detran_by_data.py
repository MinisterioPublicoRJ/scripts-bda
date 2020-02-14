from pyspark.sql.functions import col, lit, when
from pyspark.sql.functions import udf
from opg.base import spark_session
from opg.opg_utils import uuidsha, update_uuid
import sys

spark = spark_session('teste')

uuidshaudf = spark.udf.register('uuidshaudf', uuidsha)

# bases_dev_p = spark.table("bases_dev.pcivil_placas")
# data_min = bases_dev_p.select("datapassagem").agg({"datapassagem": "min"}).first()[0]
# data_min = data_min.strftime("%Y-%m-%d %H:%M")

data_min = sys.argv[1]

df = spark.table('staging.pcivil_placas')

df = df.where("""datapassagem is null or 
    datapassagem <= cast(from_unixtime(unix_timestamp('{}', 'yyyy-MM-dd HH:mm'), 'yyyy-MM-dd HH:mm:ss') as timestamp)""".format(data_min))

df = df.withColumn('uuid',
            uuidshaudf(
                col('placa'),
                col('datapassagem').cast('string'),
                col('lat').cast('string'),
                col('lon').cast('string')
            )
)

#df.write.mode("overwrite").format("parquet").saveAsTable("bases_dev.detran_regcivil_teste")
df.write.mode("overwrite").format("parquet").saveAsTable("bases_dev.pcivil_placas")

# df_2 = spark.table("bases_dev.detran_regcivil_teste")
# df_2.write.mode('append').saveAsTable("bases_dev.pcivil_placas")