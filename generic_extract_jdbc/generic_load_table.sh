#!/bin/sh

spark2-submit \
    --jars /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/ojdbc6.jar \
    --queue root.mpmapas \
    --num-executors 50 \
    --executor-cores 1 \
    --executor-memory 22g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=300 \
    --py-files src/timer.py,src/base.py,packages/*.whl,packages/*.egg \
    src/generic_load_table.py 2>> error.log
