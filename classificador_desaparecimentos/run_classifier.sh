#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --py-files base.py,packages/*.whl,packages/*.egg \
    --jars /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/ojdbc6.jar \
    predict.py 2>> error.log