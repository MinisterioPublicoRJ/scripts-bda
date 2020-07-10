#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --master yarn --deploy-mode cluster \
    --queue root.robopromotoria \
    --num-executors 10 \
    --driver-memory 5g \
    --executor-cores 5 \
    --executor-memory 8g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=3600 \
    --conf spark.speculation=true \
    --conf spark.speculation.multiplier=2 \
    --conf spark.speculation.quantile=0 \
    --conf spark.locality.wait=0 \
    --conf spark.sql.shuffle.partitions=30 \
    --conf spark.default.parallelism=30 \
    --conf spark.shuffle.io.numConnectionsPerPeer=3 \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files src/utils.py,packages/*.whl,packages/*.egg,packages/*.zip src/tabela_pip_indicadores_sucesso.py $@
