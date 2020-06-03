import datetime
import requests
import json
import os

ERROR = "ERROR"
SUCCESS = "SUCCESS"
SUCCESS_MESSAGE = "Processo {} executado com sucesso"
ERROR_MESSAGE = "Erro em {}: {}"

def send_log(message, module, levelname, solr_server):
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3].replace(" ", "T") + "Z"
    date_now = datetime.datetime.now().strftime("%Y-%m-%d")

    data = [
            {"name": os.path.basename(__main__.__file__), 
            "module": module, 
            "asctime": time_stamp, 
            "date": date_now, 
            "message": message, 
            "levelname": levelname
            }]
    params =  { "commit" : "true" }
    headers = {'content-type' : 'application/json'}
    requests.post("{}log_files/update".format(solr_server), \
        data = json.dumps(data), headers = headers, params=params)