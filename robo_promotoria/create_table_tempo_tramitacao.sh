#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --master yarn --deploy-mode cluster \
    --queue root.mpmapas \
    --num-executors 10 \
    --driver-memory 6g \
    --executor-cores 5 \
    --executor-memory 8g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.driver.maxResultSize=6000 \
    --conf spark.default.parallelism=30 \
    --conf spark.sql.shuffle.partitions=30 \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:G1HeapRegionSize=16M" \
    --py-files src/utils.py,src/files_tempo_tramitacao.zip,packages/*.whl,packages/*.egg,packages/*.zip src/tabela_tempo_tramitacao.py $@
