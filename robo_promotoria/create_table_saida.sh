#!/bin/sh

spark2-submit \
    --num-executors 10 \
    --executor-memory 10g \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=300 \
    --py-files packages/*.whl,packages/*.egg tabela_saida.py