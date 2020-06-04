import datetime
import requests
import json
import os
import sys
import pysolr

ERROR = "ERROR"
SUCCESS = "SUCCESS"
SUCCESS_MESSAGE = "Processo {} executado com sucesso"
ERROR_MESSAGE = "Erro em {}: {}"

def connect_to_solr(zookeeper_server):
    zookeeper = pysolr.ZooKeeper(zookeeper_server)
    return pysolr.SolrCloud(zookeeper, "log_files", timeout=300)

def send_data_to_solr(data, solr_server):
    solr = connect_to_solr(solr_server)
    solr.add(data, overwrite=True)

def send_log(message, module, levelname, solr_server):
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3].replace(" ", "T") + "Z"
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

# def send_log(message, module, levelname, solr_server):
#     time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3].replace(" ", "T") + "Z"
#     date_now = datetime.datetime.now().strftime("%Y-%m-%d")

#     data = [
#             {"name": os.path.basename(sys.argv[0]), 
#             "module": module, 
#             "asctime": time_stamp, 
#             "date": date_now, 
#             "message": message, 
#             "levelname": levelname
#             }]
#     params =  { "commit" : "true" }
#     headers = {'content-type' : 'application/json'}
#     requests.post("{}log_files/update".format(solr_server), \
#         data = json.dumps(data), headers = headers, params=params)