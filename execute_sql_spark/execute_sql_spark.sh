#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --master yarn --deploy-mode cluster \
    --queue root.mpmapas \
    --num-executors 3 \
    --executor-cores 1 \
    --executor-memory 2g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=300 \
    --py-files packages/*.whl,packages/*.egg,packages/*.zip src/execute_sql_spark.py $@
