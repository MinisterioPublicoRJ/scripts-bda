#!/bin/sh
export PYTHONIOENCODING=utf8

OUTPUT_TABLE_NAME="tb_tempo_tramitacao_integrado"

spark-submit --master yarn --deploy-mode cluster \
    --keytab "/home/mpmapas/keytab/mpmapas.keytab" \
    --principal mpmapas \
    --queue root.robopromotoria \
    --num-executors 12 \
    --driver-memory 6g \
    --executor-cores 5 \
    --executor-memory 10g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.driver.maxResultSize=1g \
    --conf spark.default.parallelism=30 \
    --conf spark.sql.shuffle.partitions=30 \
    --conf spark.network.timeout=3600 \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf spark.locality.wait=0 \
    --conf spark.shuffle.io.numConnectionsPerPeer=3 \
    --conf spark.yarn.appMasterEnv.PYTHON_EGG_CACHE="/tmp" \
    --conf spark.yarn.appMasterEnv.PYTHON_EGG_DIR="/tmp" \
    --conf spark.executorEnv.PYTHON_EGG_DIR="/tmp" \
    --conf spark.executorEnv.PYTHON_EGG_CACHE="/tmp" \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files ../utilities/*.py,src/files_tempo_tramitacao.zip,packages/*.whl,packages/*.egg,packages/*.zip src/tabela_tempo_tramitacao.py $@ -t ${OUTPUT_TABLE_NAME}
