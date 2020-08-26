#!/bin/sh
export PYTHONIOENCODING=utf8

spark-submit --py-files src/*.py,packages/*.whl,packages/*.egg,packages/*.zip src/change_file_hdfs_encoding.py $@
