from impala.dbapi import connect as impala_connect
import datetime
import requests
import json
import os
import sys
import pysolr
from requests_kerberos import HTTPKerberosAuth, REQUIRED 
kerberos_auth = HTTPKerberosAuth(mutual_authentication=REQUIRED, sanitize_mutual_error_response=False)

ERROR = "ERROR"
SUCCESS = "SUCCESS"
SUCCESS_MESSAGE = "Processo {} executado com sucesso"
ERROR_MESSAGE = "Erro em {}: {}"

def connect_to_solr(zookeeper_server):
    print(zookeeper_server)
    zookeeper = pysolr.ZooKeeper(zookeeper_server)
    return pysolr.SolrCloud(zookeeper, "log_files", timeout=300, auth=kerberos_auth)

def send_data_to_solr(data, solr_server):
    solr = connect_to_solr(solr_server)
    solr.add(data, overwrite=True)

def send_log(message, module, levelname, solr_server):
    #time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "T") + "Z"
    time_stamp = datetime.datetime.now().isoformat()+'Z'
    date_now = datetime.datetime.now().strftime("%Y-%m-%d")

    data = [
            {"name": os.path.basename(sys.argv[0]), 
            "module": module, 
            "asctime": time_stamp, 
            "date": date_now, 
            "message": message, 
            "levelname": levelname
            }]
    send_data_to_solr(data, solr_server)

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
