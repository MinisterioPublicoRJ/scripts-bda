#!/bin/sh
export PYTHONIOENCODING=utf8

spark2-submit --master yarn --deploy-mode cluster \
    --queue root.mpmapas \
    --num-executors 3 \
    --executor-cores 1 \
    --executor-memory 5g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=300 \
    --py-files src/utils.py,packages/*.whl,packages/*.egg,packages/*.zip src/tabela_saida.py $@
