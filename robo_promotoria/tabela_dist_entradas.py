import pyspark
from datetime import datetime
from decouple import config

from pyspark.sql.functions import (
    unix_timestamp,
    from_unixtime,
    current_timestamp,
    date_format
)

from utils import _update_impala_table

spark = pyspark.sql.session.SparkSession \
        .builder \
        .appName("criar_tabela_distribuicao_entradas") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .enableHiveSupport() \
        .getOrCreate()

schema_exadata = config('SCHEMA_EXADATA')
schema_exadata_aux = config('SCHEMA_EXADATA_AUX')

nb_past_days = config('DIST_ENTRADAS_NB_PAST_DAYS', default=60)

comb_dates = spark.sql(
    """
    SELECT date_sub(date_add(to_date(current_timestamp()), 1), nbr) as dt,
    vist_orgi_orga_dk as comb_orga_dk,
    cdmatricula as comb_cdmatricula
    FROM (
        SELECT row_number() over (order by vist_dt_abertura_vista) AS nbr
        FROM {0}.mcpr_vista 
        LIMIT {1}
    ) t1 CROSS JOIN (
        SELECT DISTINCT vist_orgi_orga_dk, cdmatricula 
        FROM {0}.mcpr_vista v
        JOIN {0}.mcpr_pessoa p ON v.VIST_PESF_PESS_DK_RESP_ANDAM = p.PESS_DK
        JOIN {0}.rh_funcionario rf ON p.PESS_ID_CADASTRO_RECEITA = rf.CPF
        WHERE to_date(VIST_DT_ABERTURA_VISTA) >= date_sub(to_date(current_timestamp()), {1})
    ) t2
    """.format(schema_exadata, nb_past_days)
)
comb_dates.registerTempTable('date_combs')

entradas_table = spark.sql(
    """
    SELECT dt,
    comb_orga_dk,
    comb_cdmatricula,
    nvl(COUNT(vist_dt_abertura_vista), 0) as nr_entradas
    FROM {0}.mcpr_vista v
    JOIN {0}.mcpr_documento ON docu_dk = vist_docu_dk
    JOIN {0}.mcpr_pessoa p ON v.VIST_PESF_PESS_DK_RESP_ANDAM = p.PESS_DK
    JOIN {0}.rh_funcionario rf ON p.PESS_ID_CADASTRO_RECEITA = rf.CPF
    RIGHT JOIN date_combs c 
        ON comb_orga_dk = vist_orgi_orga_dk
        AND comb_cdmatricula = cdmatricula
        AND dt = to_date(v.vist_dt_abertura_vista)
    WHERE (DOCU_TPST_DK != 11 OR DOCU_TPST_DK IS NULL)
    GROUP BY dt, comb_orga_dk, comb_cdmatricula
    """.format(schema_exadata)
)
entradas_table.registerTempTable('entradas_table')

estatisticas = spark.sql(
    """
    SELECT 
        t2.nr_entradas as nr_entradas_hoje, 
        t1.* 
    FROM (
        SELECT
            comb_orga_dk,
            comb_cdmatricula,
            min(nr_entradas) as minimo,
            max(nr_entradas) as maximo,
            avg(nr_entradas) as media,
            percentile(nr_entradas, 0.25) as primeiro_quartil,
            percentile(nr_entradas, 0.5) as mediana,
            percentile(nr_entradas, 0.75) as terceiro_quartil,
            percentile(nr_entradas, 0.75) - percentile(nr_entradas, 0.25) as IQR,
            percentile(nr_entradas, 0.25) - 1.5*(percentile(nr_entradas, 0.75)
                - percentile(nr_entradas, 0.25)) as Lout,
            percentile(nr_entradas, 0.75) + 1.5*(percentile(nr_entradas, 0.75)
                - percentile(nr_entradas, 0.25)) as Hout
        FROM entradas_table t
        GROUP BY comb_orga_dk, comb_cdmatricula
    ) t1
    LEFT JOIN entradas_table t2 ON t2.comb_orga_dk = t1.comb_orga_dk
    AND t2.dt = to_date(current_timestamp())
    """
).withColumn(
    "dt_inclusao",
    from_unixtime(
        unix_timestamp(current_timestamp(), 'yyyy-MM-dd'), 'yyyy-MM-dd')
    .cast('timestamp')
).withColumn("dt_partition", date_format(current_timestamp(), "ddMMyyyy"))


table_name = "{}.tb_dist_entradas".format(schema_exadata_aux)

estatisticas.write.mode("overwrite").saveAsTable(table_name)

_update_impala_table(table_name)