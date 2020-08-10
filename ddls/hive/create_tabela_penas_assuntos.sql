create external table {schema}.tb_penas_assuntos(
    hierarquia string,
    artigo_lei string,
    descricao string,
    num_docs bigint,
    max_pena double,
    multiplicador bigint,
    abuso_menor bigint,
    nome_delito string,
    id bigint
)
ROW FORMAT SERDE 
      'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES (
      'field.delim'='|',
      'separatorChar' = '|',
      'skip.header.line.count'='1',
      'serialization.encoding'='UTF-8'
)
stored as textfile
location '/user/mpmapas/staging/penas_assuntos/';