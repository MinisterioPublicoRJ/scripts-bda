#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --master yarn --deploy-mode cluster \
    --queue root.mpmapas \
    --num-executors 10 \
    --driver-memory 2g \
    --executor-cores 5 \
    --executor-memory 5g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=360 \
    --conf spark.speculation=true \
    --conf spark.speculation.quantile=0.5 \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:G1HeapRegionSize=16M" \
    --py-files src/utils.py,packages/*.whl,packages/*.egg,packages/*.zip src/tabela_dist_entradas.py $@
