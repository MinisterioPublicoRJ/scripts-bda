#!/bin/sh
export PYTHONIOENCODING=utf8

spark-submit --master yarn --deploy-mode cluster \
    --queue root.robopromotoria \
    --num-executors 12 \
    --driver-memory 6g \
    --executor-cores 5 \
    --executor-memory 18g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=3600 \
    --conf spark.locality.wait=0 \
    --conf spark.shuffle.file.buffer=1024k \
    --conf spark.io.compression.lz4.blockSize=512k \
    --conf spark.maxRemoteBlockSizeFetchToMem=1500m \
    --conf spark.reducer.maxReqsInFlight=1 \
    --conf spark.shuffle.io.maxRetries=10 \
    --conf spark.shuffle.io.retryWait=60s \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files src/utils.py,packages/*.whl,packages/*.egg,packages/*.zip src/tabela_dist_entradas.py $@
