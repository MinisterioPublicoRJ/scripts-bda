import numpy as np
import pandas as pd
import jaydebeapi as jdbc
from decouple import config
from hdfs import InsecureClient

URL_ORACLE_SERVER = config('URL_ORACLE_SERVER')
USER_ORACLE = config('USER_ORACLE')
PASSWD_ORACLE = config('PASSWD_ORACLE')
ORACLE_DRIVER_PATH = config('ORACLE_DRIVER_PATH')
ROBOT_NAME = config('ROBOT_NAME')
ROBOT_NUMBER = config('ROBOT_NUMBER')
HDFS_URL = config('HDFS_URL')
HDFS_USER = config('HDFS_USER')
HDFS_MODEL_DIR = config('HDFS_MODEL_DIR')

ORACLE_QUERY = ""

def get_results_from_hdfs(client, month):
    pass


def expand_results(data):
    pass


def get_keys(data, key_column):
    pass


def read_from_database(cursor, keys, key_column):
    pass


def get_percentage_of_change(data_hdfs, data_oracle):
    pass


def get_number_of_modifications(data_hdfs, data_oracle):
    pass


def get_number_of_final_classes(data_oracle):
    pass


def save_metrics_to_hdfs():
    pass