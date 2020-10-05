import os
import sys
import time
import datetime
import subprocess

def execute_compute_stats(table_name):

    time.sleep(7)
    
    process = subprocess.Popen(
        ['impala-shell', '-q', 'COMPUTE STATS {}'.format(table_name)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out, err = process.communicate()

    if not out:
        raise Exception(err)