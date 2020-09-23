#!/bin/sh
export PYTHONIOENCODING=utf8

OUTPUT_TABLE_NAME_ORGAO="tb_detalhe_documentos_orgao"
OUTPUT_TABLE_NAME_CPF="tb_detalhe_documentos_orgao_cpf"

spark-submit --master yarn --deploy-mode cluster \
    --keytab "/home/mpmapas/keytab/mpmapas.keytab" \
    --principal mpmapas \
    --queue root.robopromotoria \
    --num-executors 12 \
    --driver-memory 8g \
    --executor-cores 5 \
    --executor-memory 15g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=6g \
    --conf spark.driver.maxResultSize=3g \
    --conf spark.network.timeout=3600 \
    --conf spark.shuffle.io.maxRetries=5 \
    --conf spark.shuffle.io.retryWait=15s \
    --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35" \
    --py-files src/utils.py,src/files_detalhe_documentos.zip,packages/*.whl,packages/*.egg,packages/*.zip \
    src/tabela_detalhe_documento.py $@ -t1 ${OUTPUT_TABLE_NAME_ORGAO} -t2 ${OUTPUT_TABLE_NAME_CPF}

while [ $# -gt 0 ]; do

   if [[ $1 == *"-"* ]]; then
        param="${1/-/}"
        declare $param="$2"
   fi

  shift
done

impala-shell -q "INVALIDATE METADATA ${a}.${OUTPUT_TABLE_NAME_ORGAO}"
impala-shell -q "INVALIDATE METADATA ${a}.${OUTPUT_TABLE_NAME_CPF}"
