#!/bin/sh
export PYTHONIOENCODING=utf8

spark-submit --py-files packages/*.whl,packages/*.egg \
    --keytab "/home/mpmapas/keytab/mpmapas.keytab" \
    --principal mpmapas \
    --jars /home/mpmapas/libs_jars/drivers_jdbc/ojdbc6.jar \
    src/predict.py $@
