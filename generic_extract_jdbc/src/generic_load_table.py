from base import spark
from decouple import config
import params_table

url_oracle_server = config('ORACLE_SERVER')
user_oracle = config("ORACLE_USER")
passwd_oracle = config("ORACLE_PASSWORD")

config_params = params_table.params

for table in config_params['tables']:

    #Get minimum and maximum record from table oracle for just used to decide the partition stride
    query_primarykeys = " (select min({key}), max({key}) from {table_oracle}) p ".format(key=table['key_table_oracle'], table_oracle=table['table_oracle'])
    query_table = table['table_oracle']

    min_max_table = spark.read.format("jdbc") \
    .option("url", url_oracle_server) \
    .option("dbtable", query_primarykeys) \
    .option("user", user_oracle) \
    .option("password", passwd_oracle) \
    .option("driver", config_params['driver']) \
    .load()
    print('Geting min and max from table oracle')

    minimum = int(min_max_table.first()[0])
    maximum = int(min_max_table.first()[1])

    oracle_table = spark.read.format("jdbc") \
    .option("url", url_oracle_server) \
    .option("driver", config_params['driver']) \
    .option("lowerBound", minimum) \
    .option("upperBound", maximum) \
    .option("numPartitions", 50) \
    .option("partitionColumn", table['key_table_oracle']) \
    .option("dbtable", query_table) \
    .option("user", user_oracle) \
    .option("password", passwd_oracle) \
    .load()
    print('Geting all data from table oracle')

    #oracle_table.repartition(20).write.insertInto(tableName=table['table_hive'], overwrite=True)
    oracle_table.repartition(20).write.mode('overwrite').saveAsTable(table['table_hive'])
    print('Inserting data into final table %s' % table['table_hive'])

    spark.sql("ANALYZE TABLE {} COMPUTE STATISTICS".format(table['table_hive']))
    spark.sql("ANALYZE TABLE {} COMPUTE STATISTICS FOR COLUMNS".format(table['table_hive']))
