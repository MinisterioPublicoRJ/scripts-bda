#!/bin/sh

export PYTHONIOENCODING=utf8

spark-submit --master yarn --deploy-mode cluster \
    --queue root.mpmapas \
    --jars /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/ojdbc6.jar \
    --num-executors 20\
    --executor-cores 1 \
    --executor-memory 8g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.network.timeout=900 \
    --conf spark.hadoop.hive.mapred.supports.subdirectories=true \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files src/timer.py,src/base.py,packages/*.whl,packages/*.egg \
    src/save_file_pdf.py $@
