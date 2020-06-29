#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --master yarn --deploy-mode cluster \
    --queue root.mpmapas \
    --num-executors 3 \
    --driver-memory 3g \
    --executor-cores 3 \
    --executor-memory 3g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=2g \
    --conf spark.network.timeout=900 \
    --conf spark.speculation=true \
    --conf spark.locality.wait=0 \
    --conf spark.shuffle.io.numConnectionsPerPeer=2 \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files src/*.py,packages/*.egg,packages/*.whl,packages/*.zip\
    src/generate_table_civil.py $@
