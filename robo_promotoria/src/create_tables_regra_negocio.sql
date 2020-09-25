DROP TABLE IF EXISTS {schema_exadata_aux}.tb_regra_negocio_saida
---EOS---
CREATE TABLE {schema_exadata_aux}.tb_regra_negocio_saida (
   tp_andamento int
)
PARTITIONED BY (cod_atribuicao int) 
STORED AS PARQUET
---EOS---

DROP TABLE IF EXISTS {schema_exadata_aux}.tb_regra_negocio_investigacao
---EOS---
CREATE TABLE {schema_exadata_aux}.tb_regra_negocio_investigacao (
   classe_documento int
)
PARTITIONED BY (cod_atribuicao int) 
STORED AS PARQUET
---EOS---

DROP TABLE IF EXISTS {schema_exadata_aux}.tb_regra_negocio_processo
---EOS---
CREATE TABLE {schema_exadata_aux}.tb_regra_negocio_processo (
   classe_documento int
)
PARTITIONED BY (cod_atribuicao int) 
STORED AS PARQUET
---EOS---

INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao)
SELECT cldc_dk as classe_documento, cod_pct as cod_atribuicao
FROM {schema_exadata}.mcpr_classe_docto_mp
CROSS JOIN (SELECT DISTINCT cod_pct FROM {schema_exadata_aux}.atualizacao_pj_pacote) p
WHERE cldc_dk IN (51219, 51220, 51221, 51222, 51223, 392, 395)
AND cod_pct IN (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)
---EOS---
INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao)
SELECT tppr_dk as tp_andamento, cod_pct as cod_atribuicao
FROM {schema_exadata}.mcpr_tp_andamento
CROSS JOIN (SELECT DISTINCT cod_pct FROM {schema_exadata_aux}.atualizacao_pj_pacote) p
WHERE tppr_dk IN (6251,6644,6657,6326,6655)
AND cod_pct IN (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)
---EOS---
INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao)
SELECT cldc_dk as classe_documento, cod_pct as cod_atribuicao
FROM {schema_exadata}.mcpr_classe_docto_mp
CROSS JOIN (SELECT DISTINCT cod_pct FROM {schema_exadata_aux}.atualizacao_pj_pacote) p
WHERE cldc_dk IN (18, 126, 127, 159, 175, 176, 177, 319, 320, 323, 441, 582, 51205, 51217, 51218)
AND cod_pct IN (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)
---EOS---
INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao)
SELECT cldc_dk as classe_documento, cod_pct as cod_atribuicao
FROM {schema_exadata}.mcpr_classe_docto_mp
CROSS JOIN (SELECT DISTINCT cod_pct FROM {schema_exadata_aux}.atualizacao_pj_pacote) p
WHERE cldc_dk IN (3, 494, 590)
AND cod_pct IN (200, 201, 202, 203, 204, 205, 206, 207, 208, 209)
---EOS---
INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao)
SELECT tppr_dk as tp_andamento, cod_pct as cod_atribuicao FROM {schema_exadata}.mcpr_tp_andamento
CROSS JOIN (SELECT DISTINCT cod_pct FROM {schema_exadata_aux}.atualizacao_pj_pacote) p 
WHERE tppr_dk IN (
7827, 7914, 7883, 7868, 6361, 6362, 6391, 7922, 7928, 7915, 7917,
6252,6253,1201,1202,6254)
AND cod_pct IN (200, 201, 202, 203, 204, 205, 206, 207, 208, 209)
