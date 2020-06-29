CREATE TABLE acervo_painel(  dt string,   cdorgao string,   entradas double,   saidas double,   saldo string,   tipo string,   orgao string,   cluster int)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/acervo_painel') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/acervo_painel'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saldo\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='270885',   'transient_lastDdlTime'='1565195594')
CREATE TABLE atualizacao_pj_pacote(  cod_pct int,   pacote_atribuicao string,   id_orgao string,   orgao_codamp string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/atualizacao_pj_pacote') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/atualizacao_pj_pacote'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"COD_PCT\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"PACOTE_ATRIBUICAO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"ID_ORGAO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"ORGAO_CODAMP\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='13063',   'transient_lastDdlTime'='1575320754')
CREATE TABLE base_mari_25_11(  orgao string,   cdorgao double,   atribuicao string,   qtd double,   cluster int,   dt string,   entradas double,   saidas double,   tipo string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/base_mari_25_11') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/base_mari_25_11'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"atribuicao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"qtd\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='287136',   'transient_lastDdlTime'='1574714271')
CREATE TABLE painel_acervo(  orgao string,   cdorgao string,   atribuicao string,   qtd double,   cluster int,   dt string,   entradas double,   saidas double,   tipo string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_acervo') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_acervo'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"atribuicao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"qtd\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='480778',   'transient_lastDdlTime'='1565199486')
CREATE TABLE painel_mari(  dt string,   cdorgao string,   entradas double,   saidas double,   saldo string,   tipo string,   orgao string,   cluster int)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_mari') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_mari'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saldo\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='274264',   'transient_lastDdlTime'='1579115023')
CREATE TABLE painel_novembro_mari(  orgao string,   cdorgao double,   atribuicao string,   qtd double,   cluster int,   dt string,   entradas double,   saidas double,   tipo string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_novembro_mari') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_novembro_mari'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"atribuicao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"qtd\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='394327',   'transient_lastDdlTime'='1575310589')
CREATE TABLE painel_outubro(  orgao string,   cdorgao double,   atribuicao string,   qtd double,   cluster int,   dt string,   entradas double,   saidas double,   tipo string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_outubro') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_outubro'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"atribuicao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"qtd\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='395625',   'transient_lastDdlTime'='1573050823')
CREATE TABLE painel_rh_cluster_d(  cdorgao double,   orgao string,   cluster int)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_rh_cluster_d') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_rh_cluster_d'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='40162',   'transient_lastDdlTime'='1580826122')
CREATE TABLE painel_rh_cluster_total(  orgao string,   cdorgao double,   atribuicao string,   qtd double,   cluster int,   dt string,   entradas double,   saidas double,   tipo string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_rh_cluster_total') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_rh_cluster_total'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"atribuicao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"qtd\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='477355',   'transient_lastDdlTime'='1580828200')
CREATE TABLE painel_rh_crj(  orgao string,   cdorgao double,   atribuicao string,   qtd double,   cluster int,   dt string,   entradas double,   saidas double,   tipo string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_rh_crj') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_rh_crj'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"atribuicao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"qtd\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='477365',   'transient_lastDdlTime'='1580824944')
CREATE TABLE painel_setembro(  orgao string,   cdorgao double,   atribuicao string,   qtd double,   cluster int,   dt string,   entradas double,   saidas double,   tipo string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_setembro') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_setembro'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"atribuicao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"qtd\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='308635',   'transient_lastDdlTime'='1568921630')
CREATE TABLE painel_teste(  dt string,   cdorgao double,   entradas double,   saidas double,   saldo double,   tipo string,   orgao string,   cluster int)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_teste') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_teste'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saldo\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='173409',   'transient_lastDdlTime'='1574709136')
CREATE TABLE painel_teste123(  orgao string,   cdorgao double,   atribuicao string,   qtd double,   cluster int,   dt string,   entradas double,   saidas double,   tipo string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_teste123') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_teste123'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"atribuicao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"qtd\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='396274',   'transient_lastDdlTime'='1574799717')
CREATE TABLE painel_teste1234(  orgao string,   cdorgao double,   atribuicao string,   qtd double,   cluster int,   dt string,   entradas double,   saidas double,   tipo string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_teste1234') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_teste1234'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"atribuicao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"qtd\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='396274',   'transient_lastDdlTime'='1574864618')
CREATE TABLE painel_teste_crj(  orgao string,   cdorgao double,   atribuicao string,   qtd double,   cluster int,   dt string,   entradas double,   saidas double,   tipo string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_teste_crj') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/cluster.db/painel_teste_crj'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"orgao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cdorgao\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"atribuicao\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"qtd\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cluster\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"entradas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"saidas\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TIPO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='395282',   'transient_lastDdlTime'='1575309887')