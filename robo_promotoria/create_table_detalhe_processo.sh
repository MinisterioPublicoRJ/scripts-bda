#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --master yarn --deploy-mode cluster \
    --queue root.mpmapas \
    --num-executors 5 \
    --executor-cores 1 \
    --executor-memory 10g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=300 \
    --py-files src/utils.py,packages/*.whl,packages/*.egg src/tabela_detalhe_processo.py -e $SCHEMA_EXADATA -a $SCHEMA_EXADATA_AUX -i $IMPALA_HOST -o $IMPALA_PORT
