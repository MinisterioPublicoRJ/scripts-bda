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

INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (51219, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (51220, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (51221, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (51222, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (51223, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (392, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (395, 26)
---EOS---
INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (51219, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (51220, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (51221, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (51222, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (51223, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (392, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_investigacao PARTITION(cod_atribuicao) VALUES (395, 27)
---EOS---

INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (159, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (175, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (176, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (177, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (320, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (582, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (51217, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (319, 26)
---EOS---

INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (159, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (175, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (176, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (177, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (320, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (582, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (51217, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_processo PARTITION(cod_atribuicao) VALUES (319, 27)
---EOS---


INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao) VALUES (6251, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao) VALUES (6644, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao) VALUES (6657, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao) VALUES (6326, 26)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao) VALUES (6655, 26)
---EOS---

INSERT OVERWRITE TABLE {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao) VALUES (6251, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao) VALUES (6644, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao) VALUES (6657, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao) VALUES (6326, 27)
---EOS---
INSERT INTO {schema_exadata_aux}.tb_regra_negocio_saida PARTITION(cod_atribuicao) VALUES (6655, 27)
