#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --master yarn \
    --queue root.mpmapas \
    --num-executors 2 \
    --executor-cores 1 \
    --executor-memory 10g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=300 \
    --py-files src/utils.py,packages/*.whl,packages/*.egg,packages/*.zip tests/test_tabelas.py