create external table staging.pcivil_placas(
    num_camera int,
    placa string,
    lat decimal,
    lon decimal,
    datapassagem timestamp,
    velocidade decimal,
    faixa int
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'=',',
	  'timestamp.formats'='yyyy-MM-dd HH:mm:ss',
	  'skip.header.line.count'='1',
      'serialization.encoding'='LATIN1'
)
stored as textfile
location '/user/mpmapas/staging/detranbarreirasplacas/';

select * from staging.pcivil_placas;