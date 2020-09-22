#!/bin/sh
export PYTHONIOENCODING=utf8

kinit -kt /home/mpmapas/keytab/mpmapas.keytab mpmapas

spark-submit --py-files packages/*.whl,packages/*.egg \
    --keytab "/home/mpmapas/keytab/mpmapas.keytab" \
    --principal mpmapas \
    --jars /home/mpmapas/libs_jars/drivers_jdbc/ojdbc6.jar \
    src/train.py $@

kdestroy