from impala.dbapi import connect as impala_connect

def _update_impala_table(table, impalaHost, impalaPort):
    """
    Method for update table in Impala

    Parameters
    ----------
    table: string
        table name from hive

    """
    with impala_connect(
            host=impalaHost,
            port=impalaPort
    ) as conn:
        impala_cursor = conn.cursor()
        impala_cursor.execute("""
            INVALIDATE METADATA {table} """.format(table=table))