from base import spark
from decouple import config
from impala.dbapi import connect as impala_connect
import ast
import params_table
import params_table_postgre

url_oracle_server = config('ORACLE_SERVER')
user_oracle = config("ORACLE_USER")
passwd_oracle = config("ORACLE_PASSWORD")
load_all = config("LOAD_ALL")


def load_all_data(table):
    """
    Method for load all data coming from oracle table

    Parameters
    ----------
       table: dict
            "table_oracle" : oracle table name
            "pk_table_oracle" : primary key oracle table
            "update_date_table_oracle" : update date oracle table
            "table_hive" : 	hive table name
            "fields"
            (
                to use for table that
                has blob or clob columns
            ): table field names
    """

    print("Start process load all")
    # Get minimum and maximum record
    # from table oracle for just used to decide the partition stride
    query_primarykeys = """
            (select
            count(1) as total,
            min({key}),
            max({key})
            from {table_oracle}) p """.format(
                key=table['pk_table_oracle'],
                table_oracle=table['table_oracle'])

    if table.get('fields'):
        query_table = """(SELECT {fields} FROM {table_oracle}) q """.format(
            fields=table['fields'],
            table_oracle=table['table_oracle'])
    else:
        query_table = table['table_oracle']

    print('Geting min and max from table %s oracle' % table['table_oracle'])
    
    total_min_max_table = spark.read.format("jdbc") \
        .option("url", url_oracle_server) \
        .option("dbtable", query_primarykeys) \
        .option("user", user_oracle) \
        .option("password", passwd_oracle) \
        .option("driver", config_params['driver']) \
        .load()

    total = total_min_max_table.first()[0]

    if total > 0:

        minimum = int(total_min_max_table.first()[1])
        maximum = int(total_min_max_table.first()[2])

        print('Getting all data from table %s oracle' % table['table_oracle'])
        oracle_table = spark.read.format("jdbc") \
            .option("url", url_oracle_server) \
            .option("lowerBound", minimum) \
            .option("upperBound", maximum) \
            .option("numPartitions", 50) \
            .option("partitionColumn", table['pk_table_oracle']) \
            .option("dbtable", query_table) \
            .option("user", user_oracle) \
            .option("password", passwd_oracle) \
            .option("driver", config_params['driver']) \
            .load()

        table_hive = "%s.%s" % (config_params['schema_hdfs'],
                                table['table_hive'])

        print('Inserting data into final table %s' % table_hive)
        oracle_table.coalesce(20) \
            .write \
            .mode('overwrite') \
            .saveAsTable(table_hive)

        print('Update impala table %s' % table_hive)
        _update_impala_table(table_hive)

        spark.sql("ANALYZE TABLE {} COMPUTE STATISTICS".format(table_hive))


def load_part_data(table):
    """
    Method for load just the new data or updated data coming from oracle table

    Parameters
    ----------
       table: dict
            "table_oracle" : oracle table name
            "pk_table_oracle" : primary key oracle table
            "update_date_table_oracle" : update date oracle table
            "table_hive" : hive table name
            "fields"
            (
                to use for table that
                has blob or clob columns
            ): table field names

    """
    print("Start process load part data")

    # Check if table exist in hive
    spark.sql("use %s" % config_params['schema_hdfs'])
    result_table_check = spark \
        .sql("SHOW TABLES LIKE '%s'" % table['table_hive']).count()

    if result_table_check > 0:

        table_hive = "%s.%s" % (config_params['schema_hdfs'],
                                table['table_hive'])
        #spark.read.table(table_hive).createOrReplaceTempView("table_all")
        #spark.catalog.cacheTable("table_all")

        # Get count and max from hive table.
        # Count for check if table has data and max
        # for check the new data from oracle table
        total_max_table = spark \
            .sql("""
                select count(1) as total,
                max({})
                from {}
                """.format(table['pk_table_oracle'], table_hive))

        total = total_max_table.first()[0]

        if total > 0:

            max_key_value = int(total_max_table.first()[1])
            print('Getting max key value from table %s ' % table_hive)

            # If parameter update_date_table_oracle
            # exist get max update date from
            # hive table to retrive updated data from oracle table
            if table['update_date_table_oracle']:
                max_date_value = spark.sql("""
                    select max(%s)
                    from table_all """ % table['update_date_table_oracle']) \
                        .first()[0].strftime("%Y-%m-%d")

                condition = """
                or TO_CHAR({update_date_table_oracle},'YYYY-MM-DD')
                > '{max_date_value}'
                """.format(
                    update_date_table_oracle=table['update_date_table_oracle'],
                    max_date_value=max_date_value)

                print("""
                Getting max date from
                table %s and add condition to query
                """ % table_hive)

            # Get all last data inserted and all data updated in table oracle
            query = """
                    (SELECT {fields} FROM {table_oracle}
                    WHERE {key} > {max_key_value} {condition}) q
                    """.format(
                        key=table['pk_table_oracle'],
                        table_oracle=table['table_oracle'],
                        max_key_value=max_key_value,
                        fields=table['fields'] if table.get('fields') else "*",
                        condition=condition if
                        table['update_date_table_oracle'] else "")

            print("""
                Getting new data
                from table %s oracle """ % table['table_oracle'])

            spark.read.format("jdbc") \
                .option("url", url_oracle_server) \
                .option("dbtable", query) \
                .option("user", user_oracle) \
                .option("password", passwd_oracle) \
                .option("driver", config_params['driver']) \
                .load().createOrReplaceTempView("table_delta")

            total = spark.sql("select count(1) from table_delta").first()[0]

            if total > 0:
                # Join the actual data hive table
                # with the updated data to replace old data with new data

                #update_df = spark.sql("""
                #select table_all.*
                #from table_all
                #join table_delta
                #on table_all.{key} = table_delta.{key} """.format(
                #    key=table['pk_table_oracle']))

                #table_all_df = spark.sql("from table_all")
                table_delta_df = spark.sql("from table_delta")

                print("""
                Update actual data in table
                hive with new data from table oracle
                """)

                #total_df = table_all_df \
                #    .subtract(update_df) \
                #    .union(table_delta_df)

                print('Writing data in hdfs like table %s ' % table_hive)
                #total_df.write.mode("overwrite").saveAsTable("temp_table")
                #temp_table = spark.table("temp_table")
                table_delta_df.coalesce(20) \
                    .write.mode('append') \
                    .saveAsTable(table_hive)

                print('Update impala table %s' % table_hive)
                _update_impala_table(table_hive)

                #spark.sql("drop table temp_table")

            spark.catalog.clearCache()


def _update_impala_table(table):
    """
    Method for update table in Impala

    Parameters
    ----------
       table: string
            table name from hive

    """
    with impala_connect(
            host=config('IMPALA_HOST'),
            port=config('IMPALA_PORT', cast=int)
    ) as conn:
        impala_cursor = conn.cursor()
        impala_cursor.execute("""
            INVALIDATE METADATA {table} """.format(table=table))


load_all = ast.literal_eval(load_all)
config_params = params_table.params
#config_params = params_table_postgre.params

for table in config_params['tables']:
    if load_all:
        load_all_data(table)
    else:
        load_part_data(table)
