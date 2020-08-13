create external table {schema}.tb_pip_cisp(
    pip_codigo int,
    cisp_codigo int,
    cisp_nome string
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES (
      'field.delim'=',',
      'separatorChar' = ',',
	  'skip.header.line.count'='1',
      'serialization.encoding'='UTF-8'
)
stored as textfile
location '/user/mpmapas/staging/pip_cisps/';