from base import spark
import argparse
from impala.dbapi import connect as impala_connect
import ast
import params_table_promotron
import params_table_oraupsert
import params_table_oracle
import params_table_postgre
from pyspark.sql.functions import base64, col, date_format


dic_params = {
                "ORACLE_PROMOTRON" : params_table_promotron.params,
                "ORACLE_UPSERT" : params_table_oraupsert.params,
                "ORACLE": params_table_oracle.params, 
                "POSTGRE" : params_table_postgre.params
            }

def get_total_record(table):
    if table.get('no_lower_upper_bound'):
        return """
                (select
                count(1) as total
                from {table_jdbc}) p """.format(
                    table_jdbc=table['table_jdbc'])
    else:
        return """
                (select
                count(1) as total,
                min({key}),
                max({key})
                from {table_jdbc}) p """.format(
                    key=table['pk_table_jdbc'],
                    table_jdbc=table['table_jdbc'])


def load_table(table, total_min_max_table, query_table, options):
    if table.get('no_lower_upper_bound'):
        return spark.read.format("jdbc") \
            .option("url", options['jdbc_server']) \
            .option("numPartitions", 70) \
            .option("dbtable", query_table) \
            .option("user", options['jdbc_user']) \
            .option("password", options['jdbc_password']) \
            .option("driver", config_params['driver']) \
            .load()
    else:

        minimum = int(total_min_max_table.first()[1])
        maximum = int(total_min_max_table.first()[2])

        print('Getting all data from table %s jdbc' % table['table_jdbc'])
        return spark.read.format("jdbc") \
            .option("url", options['jdbc_server']) \
            .option("lowerBound", minimum) \
            .option("upperBound", maximum) \
            .option("numPartitions", 70) \
            .option("partitionColumn", table['pk_table_jdbc']) \
            .option("dbtable", query_table) \
            .option("user", options['jdbc_user']) \
            .option("password", options['jdbc_password']) \
            .option("driver", config_params['driver']) \
            .load()


def load_all_data(table, options):
    """
    Method for load all data coming from jdbc table

    Parameters
    ----------
    table: dict
        "table_jdbc" : jdbc table name
        "pk_table_jdbc" : primary key jdbc table
        "update_date_table_jdbc" : update date jdbc table
        "table_hive" : 	hive table name
        "fields"
        (
            to use for table that
            has blob or clob columns
        ): table field names
    options: dict
        All parameters from JDBC
    """

    print("Start process load all")
    # Get minimum and maximum record
    # from table jdbc for just used to decide the partition stride
    query_primarykeys = get_total_record(table)

    if table.get('fields'):
        query_table = """(SELECT {fields} FROM {table_jdbc}) q """.format(
            fields=table['fields'],
            table_jdbc=table['table_jdbc'])
    else:
        query_table = table['table_jdbc']

    print('Geting min and max from table %s jdbc' % table['table_jdbc'])
    
    total_min_max_table = spark.read.format("jdbc") \
        .option("url", options['jdbc_server']) \
        .option("dbtable", query_primarykeys) \
        .option("user", options['jdbc_user']) \
        .option("password", options['jdbc_password']) \
        .option("driver", config_params['driver']) \
        .load()

    total = total_min_max_table.first()[0]

    if total > 0:

        jdbc_table = load_table(table, total_min_max_table, query_table, options)

        table_hive = "%s.%s" % (options['schema_exadata'],
                                table['table_hive'])

        print('Inserting data into final table %s' % table_hive)

        final_df = transform_col_binary(jdbc_table)

        if table.get('partition_column') and table.get('date_partition_format'):
            final_df = final_df.withColumn("year_month",
                date_format(table['partition_column'], table['date_partition_format']).cast("int"))

            final_df.write.partitionBy('year_month') \
                .mode("overwrite") \
                .saveAsTable(table_hive)

        elif table.get('partition_column'):
            final_df.write.partitionBy(table['partition_column']) \
                .mode("overwrite") \
                .saveAsTable(table_hive)
        else:
            final_df \
                .write \
                .mode('overwrite') \
                .saveAsTable(table_hive)

        print('Update impala table %s' % table_hive)
        _update_impala_table(table_hive, options)

        spark.sql("ANALYZE TABLE {} COMPUTE STATISTICS".format(table_hive))


def load_part_data(table, options):
    """
    Method for load just the new data or updated data coming from jdbc table

    Parameters
    ----------
    table: dict
        "table_jdbc" : jdbc table name
        "pk_table_jdbc" : primary key jdbc table
        "update_date_table_jdbc" : update date jdbc table
        "table_hive" : hive table name
        "fields"
        (
            to use for table that
            has blob or clob columns
        ): table field names
    options: dict
        All parameters from JDBC
    """
    print("Start process load part data")

    # Check if table exist in hive
    spark.sql("use %s" % options['schema_exadata'])
    result_table_check = spark \
        .sql("SHOW TABLES LIKE '%s'" % table['table_hive']).count()

    if result_table_check > 0 and not table.get('no_partition_column'):

        table_hive = "%s.%s" % (options['schema_exadata'],
                                table['table_hive'])

        # Get count and max from hive table.
        # Count for check if table has data and max
        # for check the new data from jdbc table
        total_max_table = spark \
            .sql("""
                select count(1) as total,
                max({})
                from {}
                """.format(table['pk_table_jdbc'], table_hive))

        total = total_max_table.first()[0]

        if total > 0:

            max_key_value = int(total_max_table.first()[1])
            print('Getting max key value from table %s ' % table_hive)

            # If parameter update_date_table_jdbc
            # exist get max update date from
            # hive table to retrive updated data from jdbc table
            if table['update_date_table_jdbc']:
                max_date_value = spark.sql("""
                    select max(%s)
                    from table_all """ % table['update_date_table_jdbc']) \
                        .first()[0].strftime("%Y-%m-%d")

                condition = """
                or TO_CHAR({update_date_table_jdbc},'YYYY-MM-DD')
                > '{max_date_value}'
                """.format(
                    update_date_table_jdbc=table['update_date_table_jdbc'],
                    max_date_value=max_date_value)

                print("""
                Getting max date from
                table %s and add condition to query
                """ % table_hive)

            # Get all last data inserted and all data updated in table jdbc
            query = """
                    (SELECT {fields} FROM {table_jdbc}
                    WHERE {key} > {max_key_value} {condition}) q
                    """.format(
                        key=table['pk_table_jdbc'],
                        table_jdbc=table['table_jdbc'],
                        max_key_value=max_key_value,
                        fields=table['fields'] if table.get('fields') else "*",
                        condition=condition if
                        table['update_date_table_jdbc'] else "")

            print("""
                Getting new data
                from table %s jdbc """ % table['table_jdbc'])

            spark.read.format("jdbc") \
                .option("url", options['jdbc_server']) \
                .option("dbtable", query) \
                .option("user", options['jdbc_user']) \
                .option("password", options['jdbc_password']) \
                .option("driver", config_params['driver']) \
                .load().createOrReplaceTempView("table_delta")

            total = spark.sql("select count(1) from table_delta").first()[0]

            if total > 0:
                # Join the actual data hive table
                # with the updated data to replace old data with new data

                table_delta_df = spark.sql("from table_delta")

                print("""
                Update actual data in table
                hive with new data from table jdbc
                """)

                print('Writing data in hdfs like table %s ' % table_hive)

                final_df = transform_col_binary(table_delta_df)
                final_df.coalesce(1) \
                    .write.mode('append') \
                    .saveAsTable(table_hive)

                print('Update impala table %s' % table_hive)
                _update_impala_table(table_hive, options)

            spark.catalog.clearCache()


def _update_impala_table(table, options):
    """
    Method for update table in Impala

    Parameters
    ----------
    table: string
        table name from hive

    """
    with impala_connect(
            host=options['impala_host'],
            port=int(options['impala_port'])
    ) as conn:
        impala_cursor = conn.cursor()
        impala_cursor.execute("""
            INVALIDATE METADATA {table} """.format(table=table))
        impala_cursor.execute("""
            COMPUTE STATS {table} """.format(table=table))

def transform_col_binary(data_frame):
    """
    Method for transform column binary to base64

    Parameters
    ----------
    data_frame: DataFrame
        dataframe to be transformed
        
    Returns
    -------
    dataframe
        A transformed dataframe with new column base64

    """
    return reduce(lambda df, (col_name, dtype): df
            .withColumn(col_name, base64(col(col_name)))
            .withColumnRenamed(col_name, 'BASE64_' + col_name)
            if dtype == 'binary' else df.withColumn(col_name, col(col_name)),
            data_frame.dtypes, data_frame)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load all table from same JDBC connection")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-s','--jdbcServer', metavar='jdbcServer', type=str, help='')
    parser.add_argument('-u','--jdbcUser', metavar='jdbcUser', type=str, help='')
    parser.add_argument('-p','--jdbcPassword', metavar='jdbcPassword', type=str, help='')
    parser.add_argument('-t','--typeJdbc', metavar='typeJdbc', type=str, help='')
    parser.add_argument('-l','--loadAll', metavar='loadAll', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    args = parser.parse_args()

    options = {
                'schema_exadata': args.schemaExadata, 
                'jdbc_server' : args.jdbcServer,
                'jdbc_user' : args.jdbcUser,
                'jdbc_password' : args.jdbcPassword,
                'type_jdbc' : args.typeJdbc,
                'load_all' : args.loadAll,
                'impala_host' : args.impalaHost,
                'impala_port' : args.impalaPort
            }

    load_all = ast.literal_eval(options['load_all'])
    config_params = dic_params[args.typeJdbc.upper()]

    for table in config_params['tables']:
        if load_all:
            load_all_data(table, options)
        else:
            load_part_data(table, options)
