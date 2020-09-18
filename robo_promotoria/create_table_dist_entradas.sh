#!/bin/sh
export PYTHONIOENCODING=utf8

OUTPUT_TABLE_NAME="tb_dist_entradas"

spark-submit --master yarn --deploy-mode cluster \
    --keytab mpmapas.keytab \
    --principal mpmapas \
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
    --py-files src/utils.py,packages/*.whl,packages/*.egg,packages/*.zip src/tabela_dist_entradas.py $@ -t ${OUTPUT_TABLE_NAME}

while [ $# -gt 0 ]; do

   if [[ $1 == *"-"* ]]; then
        param="${1/-/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi

  shift
done

kinit -kt mpmapas.keytab mpmapas
impala-shell -q "INVALIDATE METADATA ${a}.${OUTPUT_TABLE_NAME}"
kdestroy
