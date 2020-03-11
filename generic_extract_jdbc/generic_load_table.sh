#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --master yarn --deploy-mode cluster \
    --jars /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/ojdbc6.jar,/opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/postgresql-9.0-801.jdbc4.jar \
    --queue root.mpmapas \
    --num-executors 50 \
    --executor-cores 1 \
    --executor-memory 22g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=300 \
    --py-files src/*.py,packages/*.whl,packages/*.egg,packages/*.zip \
    src/generic_load_table.py -e $SCHEMA_EXADATA -s $JDBC_SERVER -u $JDBC_USER -p $JDBC_PASSWORD -t $TYPE_JDBC -l $LOAD_ALL -i $IMPALA_HOST -o $IMPALA_PORT
