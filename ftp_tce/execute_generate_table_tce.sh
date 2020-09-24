#!/bin/sh
export PYTHONIOENCODING=utf8

spark-submit --master yarn --deploy-mode cluster \
    --principal mpmapas \
    --keytab "/home/mpmapas/keytab/mpmapas.keytab" \
    --jars /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/ojdbc6.jar,/opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/postgresql-9.0-801.jdbc4.jar \
    --queue root.mpmapas \
    --num-executors 3 \
    --executor-cores 1 \
    --executor-memory 5g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=900 \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf spark.executorEnv.PYTHON_EGG_CACHE="/tmp" \
    --conf spark.executorEnv.PYTHON_EGG_DIR="/tmp" \
    --conf spark.driverEnv.PYTHON_EGG_CACHE="/tmp" \
    --conf spark.driverEnv.PYTHON_EGG_DIR="/tmp" \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files src/*.py,packages/*.whl,packages/*.egg,packages/*.zip src/generate_table_tce.py $@