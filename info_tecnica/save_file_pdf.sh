#!/bin/sh

export myDependencyJarDir=/opt/cloudera/parcels/CDH/jars
export myDependencyJarFiles=$(find $myDependencyJarDir -name 'ojdbc*.jar' | sort | tr '\n' ',' | head -c -1)

spark2-submit \
    --jars $myDependencyJarFiles \
    --num-executors 50 \
    --executor-cores 1 \
    --executor-memory 8g \
    --conf spark.hadoop.hive.mapred.supports.subdirectories=true \
    --py-files src/timer.py,src/base.py,packages/*.whl,packages/*.egg \
    src/save_file_pdf.py
