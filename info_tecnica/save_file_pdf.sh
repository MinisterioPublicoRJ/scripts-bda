#!/bin/sh

export myDependencyJarDir=/opt/cloudera/parcels/CDH/jars
export myDependencyJarFiles=$(find $myDependencyJarDir -name 'ojdbc*.jar' | sort | tr '\n' ',' | head -c -1)

spark2-submit \
    --jars $myDependencyJarFiles \
    --num-executors 50 \
    --executor-cores 1 \
    --executor-memory 8g \
    --py-files timer.py,base.py,packages/*.whl,packages/*.egg \
    save_file_pdf.py
