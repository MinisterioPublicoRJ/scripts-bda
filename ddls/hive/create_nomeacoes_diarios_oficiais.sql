SET hive.mapred.supports.subdirectories=TRUE;
SET mapred.input.dir.recursive=TRUE;

drop table staging.nomeacoes_diarios_oficiais;

create external table staging.nomeacoes_diarios_oficiais(
    acao string,
    dt_do timestamp,
    url_do string,
    dt_inicio_validade timestamp,
    nm_assessor string,
    texto_original string,
    nr_pagina string,
    nm_assessor_anterior string,
    local_cargo_anterior string,
    matricula string,
    cargo_nome string,
    local_cargo string,
    local_cargo_completo string,
    fonte_do string,
    cargo_simbolo string,
    nm_arquivo string,
    nm_deputado_urna_encontrado string,
    cpf_deputado string,
    cpf_assessor string
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.OpenCSVSerde' 
WITH SERDEPROPERTIES ( 
	  'timestamp.formats'='yyyy-MM-dd',
	  'skip.header.line.count'='1',
      'serialization.encoding'='UTF-8'
)
stored as textfile
location '/user/mpmapas/staging/nomeacoes/';

select * from staging.nomeacoes_diarios_oficiais;