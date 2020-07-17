#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --master yarn --deploy-mode cluster \
    --queue root.robopromotoria \
    --num-executors 12 \
    --driver-memory 2g \
    --executor-cores 5 \
    --executor-memory 10g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=3600 \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files src/utils.py,packages/*.whl,packages/*.egg,packages/*.zip src/tabela_detalhe_documento.py $@
