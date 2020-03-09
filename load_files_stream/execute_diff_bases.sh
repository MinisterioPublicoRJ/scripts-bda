spark2-submit --queue root.mpmapas \
    --num-executors 20 \
    --executor-cores 1 \
    --executor-memory 2g \
    --conf spark.debug.maxToStringFields=2000 \
    --conf spark.executor.memoryOverhead=4096 \
    --conf spark.network.timeout=300 \
	--py-files packages/*.whl,packages/*.egg src/diff_bases_placas_detran_by_data.py '2020-02-14 14:06'
