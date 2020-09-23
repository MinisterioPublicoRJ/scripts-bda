#!/bin/sh
export PYTHONIOENCODING=utf8

OUTPUT_TABLE_NAME_PROCEDIMENTOS="tb_pip_investigados_procedimentos"
OUTPUT_TABLE_NAME_INVESTIGADOS="tb_pip_investigados"
OUTPUT_TABLE_DATE_CHECKED="dt_checked_investigados"

spark-submit --master yarn --deploy-mode cluster \
    --keytab "/home/mpmapas/keytab/mpmapas.keytab" \
    --principal mpmapas \
    --conf spark.executorEnv.PYTHON_EGG_CACHE="/tmp" \
    --conf spark.executorEnv.PYTHON_EGG_DIR="/tmp" \
    --conf spark.driverEnv.PYTHON_EGG_CACHE="/tmp" \
    --conf spark.driverEnv.PYTHON_EGG_DIR="/tmp" \
    --queue root.robopromotoria \
    --num-executors 12 \
    --driver-memory 6g \
    --executor-cores 5 \
    --executor-memory 10g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=3600 \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf spark.locality.wait=0 \
    --conf spark.shuffle.io.numConnectionsPerPeer=3 \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files src/utils.py,packages/*.whl,packages/*.egg,packages/*.zip,packages/*.py \
    src/tabela_pip_investigados.py $@ -t1 ${OUTPUT_TABLE_NAME_PROCEDIMENTOS} -t2 ${OUTPUT_TABLE_NAME_INVESTIGADOS} -t3 ${OUTPUT_TABLE_DATE_CHECKED}

while [ $# -gt 0 ]; do

   if [[ $1 == *"-"* ]]; then
        param="${1/-/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi

  shift
done

impala-shell -q "INVALIDATE METADATA ${a}.${OUTPUT_TABLE_NAME_PROCEDIMENTOS}"
impala-shell -q "INVALIDATE METADATA ${a}.${OUTPUT_TABLE_NAME_INVESTIGADOS}"
impala-shell -q "INVALIDATE METADATA ${a}.${OUTPUT_TABLE_DATE_CHECKED}"
