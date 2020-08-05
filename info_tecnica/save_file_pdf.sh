#!/bin/sh

export myDependencyJarDir=/opt/cloudera/parcels/CDH/jars
export myDependencyJarFiles=$(find $myDependencyJarDir -name 'ojdbc*.jar' | sort | tr '\n' ',' | head -c -1)
export PYTHONIOENCODING=utf8

spark-submit --master yarn --deploy-mode cluster \
    --queue root.mpmapas \
    --jars $myDependencyJarFiles \
    --num-executors 20\
    --executor-cores 1 \
    --executor-memory 8g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.network.timeout=900 \
    --conf spark.hadoop.hive.mapred.supports.subdirectories=true \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files src/timer.py,src/base.py,packages/*.whl,packages/*.egg \
    src/save_file_pdf.py $@
