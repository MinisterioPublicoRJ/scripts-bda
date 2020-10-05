import os
import sys
import json
import time
import pysolr
import datetime
import requests
import subprocess

from requests_kerberos import HTTPKerberosAuth, REQUIRED 
from impala.dbapi import connect as impala_connect

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

def execute_compute_stats(table_name):

    time.sleep(7)
    
    process = subprocess.Popen(
        ['impala-shell', '-q', 'COMPUTE STATS {}'.format(table_name)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out, err = process.communicate()

    if not out:
        raise Exception(err)