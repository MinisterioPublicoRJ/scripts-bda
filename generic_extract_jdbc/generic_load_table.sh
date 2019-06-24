#!/bin/sh

spark2-submit \
    --name "GENERIC_LOAD_TABLE" \
    --jars /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/ojdbc6.jar \
    --queue root.mpmapas \
    --num-executors 200 \
    --executor-cores 1 \
    --executor-memory 8g \
    --py-files src/timer.py,src/base.py,packages/*.whl,packages/*.egg \
   src/generic_load_table.py 2>> error.log
