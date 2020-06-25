CREATE TABLE tb_regcivil(  base varchar(50),   nu_rg varchar(50),   dt_expedicao_carteira timestamp,   no_cidadao varchar(255),   no_paicidadao varchar(255),   no_maecidadao varchar(255),   naturalidade varchar(100),   dt_nascimento timestamp,   documento_origem varchar(255),   nu_cpf varchar(11),   endereco varchar(255),   bairro varchar(255),   municipio varchar(255),   uf varchar(2),   cep varchar(8))PARTITIONED BY (   year int)ROW FORMAT SERDE   'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' STORED AS INPUTFORMAT   'org.apache.hadoop.mapred.TextInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/detran.db/tb_regcivil'TBLPROPERTIES (  'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.schema.numPartCols'='1',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"base\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(50)\"}},{\"name\":\"nu_rg\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(50)\"}},{\"name\":\"dt_expedicao_carteira\",\"type\":\"timestamp\",\"nullable\":true,\"metadata\":{}},{\"name\":\"no_cidadao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(255)\"}},{\"name\":\"no_paicidadao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(255)\"}},{\"name\":\"no_maecidadao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(255)\"}},{\"name\":\"naturalidade\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(100)\"}},{\"name\":\"dt_nascimento\",\"type\":\"timestamp\",\"nullable\":true,\"metadata\":{}},{\"name\":\"documento_origem\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(255)\"}},{\"name\":\"nu_cpf\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(11)\"}},{\"name\":\"endereco\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(255)\"}},{\"name\":\"bairro\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(255)\"}},{\"name\":\"municipio\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(255)\"}},{\"name\":\"uf\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(2)\"}},{\"name\":\"cep\",\"type\":\"string\",\"nullable\":true,\"metadata\":{\"HIVE_TYPE_STRING\":\"varchar(8)\"}},{\"name\":\"year\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}',   'spark.sql.sources.schema.partCol.0'='year',   'transient_lastDdlTime'='1591466410')
CREATE VIEW vw_regcivil AS SELECT orig.* FROM detran.tb_regcivil orig INNER JOIN (SELECT nu_rg, max(dt_expedicao_carteira) ultima_expedicao FROM detran.tb_regcivil GROUP BY nu_rg) sub ON orig.nu_rg = sub.nu_rg AND orig.dt_expedicao_carteira = sub.ultima_expedicao ORDER BY year DESC
