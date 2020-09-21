#!/bin/sh
export PYTHONIOENCODING=utf8

kinit mpmapas@BDA.LOCAL -kt ~/keytab/mpmapas.keytab
/opt/cloudera/parcels/Anaconda-5.0.1/bin/python src/change_file_hdfs_encoding.py $@