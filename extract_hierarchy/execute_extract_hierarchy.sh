#!/bin/sh
export PYTHONIOENCODING=utf8

spark-submit --master yarn --deploy-mode cluster \
    --keytab "/home/mpmapas/keytab/mpmapas.keytab" \
    --principal mpmapas \
    --queue root.mpmapas \
    --num-executors 5 \
    --driver-memory 2g \
    --executor-cores 1 \
    --executor-memory 10g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=900 \
    --conf spark.speculation=true \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf spark.locality.wait=0 \
    --conf spark.shuffle.io.numConnectionsPerPeer=3 \
    --conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE="/tmp" \
    --conf spark.yarn.appMasterEnv.PYTHON_EGG_DIR="/tmp" \
    --conf spark.executorEnv.PYTHON_EGG_DIR="/tmp" \
    --conf spark.executorEnv.PYTHON_EGG_CACHE="/tmp" \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files ../utilities/*.py,packages/*.whl,packages/*.egg,packages/*.zip src/extractor.py $@
