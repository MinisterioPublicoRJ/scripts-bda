import sys
import os
import pysolr
import datetime
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