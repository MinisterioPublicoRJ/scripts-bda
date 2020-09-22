#!/bin/sh
export PYTHONIOENCODING=utf8

spark-submit --py-files packages/*.whl,packages/*.egg \
    --keytab "/home/mpmapas/keytab/mpmapas.keytab" \
    --principal mpmapas \
    --jars /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/ojdbc6.jar \
    src/evaluate.py $@
