from base import spark
from decouple import config
import params_table


url_oracle_server = config('ORACLE_SERVER')
user_oracle = config("ORACLE_USER")
passwd_oracle = config("ORACLE_PASSWORD")
table_oracle_param = config('TABLE_ORACLE')
table_hive_param = config('TABLE_HIVE')

config_params = params_table.params

#Get minimum and maximum record from table oracle for just used to decide the partition stride
query_primarykeys = " (select min({key}), max({key}) from {table_oracle}) p ".format(key=config_params['key_table'], table_oracle=config_params['table_oracle'])

if config_params['columns_table_stg']:
    columns = ",".join(config_params['columns_table_stg'])
    query_table = " ( SELECT {qr} FROM {table_oracle} ) q ".format(qr=columns, table_oracle=config_params['table_oracle'])
else:
    query_table = config_params['table_oracle']


min_max_table = spark.read.format("jdbc") \
.option("url", url_oracle_server) \
.option("dbtable", query_primarykeys) \
.option("user", user_oracle) \
.option("password", passwd_oracle) \
.option("driver", "oracle.jdbc.driver.OracleDriver") \
.load()

minimum = int(min_max_table.first()[0])
maximum = int(min_max_table.first()[1])

oracle_table = spark.read.format("jdbc") \
.option("url", url_oracle_server) \
.option("driver", "oracle.jdbc.driver.OracleDriver") \
.option("lowerBound", minimum) \
.option("upperBound", maximum) \
.option("numPartitions", 200) \
.option("partitionColumn", config_params['key_table']) \
.option("dbtable", query_table) \
.option("user", user_oracle) \
.option("password", passwd_oracle) \
.load()

final_df = oracle_table.repartition(10).cache()

final_df.write.insertInto(tableName=config_params['table_hive_stg'], overwrite=True)

if config_params['columns_table']:
    final_df.select(config_params['columns_table']).write.insertInto(tableName=config_params['table_hive'], overwrite=True)
else:
    final_df.write.insertInto(tableName=config_params['table_hive'], overwrite=True)

spark.sql("ANALYZE TABLE {} COMPUTE STATISTICS".format(config_params['table_hive']))
spark.catalog.clearCache()
