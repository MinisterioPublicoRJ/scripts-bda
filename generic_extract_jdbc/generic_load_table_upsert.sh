#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit \
    --jars /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/ojdbc6.jar,/opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/postgresql-9.0-801.jdbc4.jar \
    --queue root.mpmapas \
    --num-executors 5 \
    --executor-cores 1 \
    --executor-memory 2g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=300 \
    --py-files /home/mpmapas/scripts-bda/generic_extract_jdbc/src/timer.py,/home/mpmapas/scripts-bda/generic_extract_jdbc/src/base.py,/home/mpmapas/scripts-bda/generic_extract_jdbc/packages/*.whl,/home/mpmapas/scripts-bda/generic_extract_jdbc/packages/*.egg \
    /home/mpmapas/scripts-bda/generic_extract_jdbc/src/generic_load_table.py
