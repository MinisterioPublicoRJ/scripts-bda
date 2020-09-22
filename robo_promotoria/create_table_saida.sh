#!/bin/sh
export PYTHONIOENCODING=utf8

OUTPUT_TABLE_NAME="tb_saida"

spark-submit --master yarn --deploy-mode cluster \
    --keytab mpmapas.keytab \
    --principal mpmapas \
    --queue root.robopromotoria \
    --num-executors 12 \
    --driver-memory 3g \
    --executor-cores 5 \
    --executor-memory 10g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=2024 \
    --conf spark.network.timeout=3600 \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf spark.locality.wait=0 \
    --conf spark.shuffle.io.numConnectionsPerPeer=3 \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files src/utils.py,packages/*.whl,packages/*.egg,packages/*.zip src/tabela_saida.py $@ -t ${OUTPUT_TABLE_NAME}

while [ $# -gt 0 ]; do

   if [[ $1 == *"-"* ]]; then
        param="${1/-/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi

  shift
done

impala-shell -q "INVALIDATE METADATA ${a}.${OUTPUT_TABLE_NAME}"
