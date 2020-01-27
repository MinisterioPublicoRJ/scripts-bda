from decouple import config
from impala.dbapi import connect as impala_connect

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