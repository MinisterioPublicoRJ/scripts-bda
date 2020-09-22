#!/bin/sh
export PYTHONIOENCODING=utf8

/opt/cloudera/parcels/Anaconda-5.0.1/bin/python src/change_file_hdfs_encoding.py $@
