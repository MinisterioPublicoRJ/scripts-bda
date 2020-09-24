#!/bin/sh
export PYTHONIOENCODING=utf8

spark-submit \
    --keytab /home/mpmapas/keytab/mpmapas.keytab \
    --principal mpmapas \
    --jars /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/ojdbc6.jar,/opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/postgresql-9.0-801.jdbc4.jar \
    --queue root.upsert \
    --num-executors 10 \
    --executor-cores 1 \
    --executor-memory 2g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=900 \
    --conf spark.speculation=true \
    --conf spark.speculation.quantile=0.5 \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:G1HeapRegionSize=16M" \
    --py-files src/*.py,packages/*.whl,packages/*.egg,packages/*.zip \
    src/generic_load_table.py $@
