from base import spark
from decouple import config
from datetime import date
import ast
import params_table

url_oracle_server = config('ORACLE_SERVER')
user_oracle = config("ORACLE_USER")
passwd_oracle = config("ORACLE_PASSWORD")
load_all = config("LOAD_ALL")


def load_all_data(table):
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
    print('Getting all data from table oracle')

    #oracle_table.repartition(20).write.insertInto(tableName=table['table_hive'], overwrite=True)
    oracle_table.repartition(20).write.mode('overwrite').saveAsTable(table['table_hive'])
    print('Inserting data into final table %s' % table['table_hive'])

    spark.sql("ANALYZE TABLE {} COMPUTE STATISTICS".format(table['table_hive']))
    #spark.sql("ANALYZE TABLE {} COMPUTE STATISTICS FOR COLUMNS".format(table['table_hive']))


def load_part_data(table):
    
    spark.read.table(table['table_hive']).createOrReplaceTempView("table_all")
    spark.catalog.cacheTable("table_all")

    max_key_value = int(spark.sql("select max(%s) from table_all" % table['key_table_oracle']).first()[0])
    print('Getting max key value from table %s ' % table['table_hive'])

    if table['update_date_table_oracle']:
        max_date_value = spark.sql("select max(%s) from table_all" % table['update_date_table_oracle']).first()[0].strftime("%Y-%m-%d")
        condition = """  
                    or TO_CHAR({update_date_table_oracle},'YYYY-MM-DD') > '{max_date_value}' 
                    """.format(update_date_table_oracle=table['update_date_table_oracle'], max_date_value=max_date_value)
        print('Getting max date from table %s and add condition to query' % table['table_hive'])
    
    #Get all last data inserted and all data updated in table oracle
    query =  """
            (SELECT * FROM {table_oracle} 
            WHERE {key} > {max_key_value} {condition}) q 
            """.format(key=table['key_table_oracle'], table_oracle=table['table_oracle'], 
                max_key_value=max_key_value, condition=condition if table['update_date_table_oracle'] else "")

    spark.read.format("jdbc") \
        .option("url", url_oracle_server) \
        .option("dbtable", query) \
        .option("user", user_oracle) \
        .option("password", passwd_oracle) \
        .option("driver", config_params['driver']) \
        .load().createOrReplaceTempView("table_delta")
    print('Getting new data from table %s oracle' % table['update_date_table_oracle'])

    #Join the actual data table hive with the updated data
    update_df = spark.sql("select table_all.* from table_all join table_delta on table_all.{key} = table_delta.{key}".format(key=table['key_table_oracle']))
    table_all_df = spark.sql("from table_all")
    table_delta_df = spark.sql("from table_delta")

    total_df = table_all_df.subtract(update_df).union(table_delta_df)
    print('Update actual data in table hive with new data from table oracle')

    total_df.write.mode("overwrite").saveAsTable("temp_table")
    temp_table = spark.table("temp_table")
    temp_table.repartition(20).write.mode('overwrite').saveAsTable(table['table_hive'])
    print('Writing data in hdfs like table %s ' % table['table_hive'])
    
    spark.sql("drop table temp_table")
    spark.catalog.clearCache()


load_all = ast.literal_eval(load_all)
config_params = params_table.params

for table in config_params['tables']:
    if load_all:
        load_all_data(table)
    else:
        load_part_data(table)
