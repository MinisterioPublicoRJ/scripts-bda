create external table saude.stg_escalacolaboradores(
    unidade_de_atendimento int,
    tipo_da_unidade string,
    especialidade string,
    setor string,
    profissional string,
    inicio timestamp,
    fim timestamp
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'=',',
	  'timestamp.formats'='dd/MM/yyyy HH:mm:ss',
	  'skip.header.line.count'='1'
)
stored as textfile
location '/user/mpmapas/staging/saude/escalacolaboradores/';


create external table saude.stg_despesasunidades(
    unidade_de_atendimento int,
    tipo_da_unidade string,
    familia_despesa string,
    item_despesa string,
    fornecedor string,
    `data` string,
    valor decimal
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'=',',
	  'skip.header.line.count'='1'
)
stored as textfile
location '/user/mpmapas/staging/saude/despesasunidades/';


unidade_de_atendimento,tipo_da_unidade,categoria_do_atendimento,data,quantidade,meta_contratual,meta
create external table saude.stg_producaoassistencial(
    unidade_de_atendimento int,
    tipo_da_unidade string,
    categoria_do_atendimento string,
    `data` string,
    quantidade bigint,
    meta_contratual int,
    meta int
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'=',',
	  'skip.header.line.count'='1'
)
stored as textfile
location '/user/mpmapas/staging/saude/producaoassistencial/';