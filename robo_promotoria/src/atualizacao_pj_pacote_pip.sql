INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote 
SELECT 
200 as cod_pct,
"PIPs Territoriais 1a CI" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao 
FROM (
    SELECT stack(24
    , 29934303
    , 29934337
    , 29926583
    , 29926616
    , 29926805
    , 29927047
    , 29933374
    , 29933418
    , 29933469
    , 29933470
    , 29933490
    , 29933502
    , 29933521
    , 29933590
    , 29933830
    , 29933850
    , 29933955
    , 29933967
    , 29933988
    , 29934004
    , 29934012
    , 29934277
    , 29934363
    , 29934376) AS (pip_codigo) 
    ) t 
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo 
---EOS---

INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote 
SELECT 
201 as cod_pct,
"PIPs Territoriais 2a CI" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao 
FROM (
    SELECT stack(5
    , 30061694
    , 30061723
    , 30061624
    , 30034384
    , 30061094) AS (pip_codigo)
    ) t 
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
---EOS---

INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote 
SELECT 
202 as cod_pct,
"PIPs Territoriais 3a CI" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao 
FROM (
    SELECT stack(9
    , 30069669
    , 30069693
    , 30069732
    , 30070041
    , 30069167
    , 30069433
    , 30069453
    , 30069490
    , 30069516) AS (pip_codigo)
    ) t 
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
---EOS---

INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote 
SELECT 
203 as cod_pct,
"PIPs Territoriais Interior" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao 
FROM (
    SELECT stack(10
    , 300977
    , 300987
    , 400731
    , 400758
    , 400736
    , 1249804
    , 5679985
    , 6115386
    , 7419262
    , 7419344) AS (pip_codigo)
    ) t 
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
---EOS---

INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote 
SELECT 
204 as cod_pct,
"PIPs Violência Doméstica 1a CI" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao 
FROM (
    SELECT stack(4
    , 29941071
    , 29941061
    , 29934401
    , 29934732) AS (pip_codigo)
    ) t 
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
---EOS---

INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote 
SELECT 
205 as cod_pct,
"PIPs Violência Doméstica 2a CI" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao 
FROM (
    SELECT stack(2
    , 30034671
    , 30061329) AS (pip_codigo)
    ) t 
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
---EOS---

INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote 
SELECT 
206 as cod_pct,
"PIPs Violência Doméstica 3a CI" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao 
FROM (
    SELECT stack(3
    , 29941322
    , 30034664
    , 30069601) AS (pip_codigo)
    ) t 
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
---EOS---

INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote 
SELECT 
207 as cod_pct,
"PIPs Especializadas 1a CI" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao 
FROM (
    SELECT stack(4
    , 29941099
    , 29941140
    , 29941222
    , 29941251) AS (pip_codigo)
    ) t 
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
---EOS---

INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote 
SELECT 
208 as cod_pct,
"PIPs Especializadas 2a CI" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao 
FROM (
    SELECT stack(2
    , 29941368
    , 30061803) AS (pip_codigo)
    ) t 
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
---EOS---

INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote 
SELECT 
209 as cod_pct,
"PIPs Especializadas 3a CI" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao 
FROM (
    SELECT stack(2
    , 30070783
    , 30070806) AS (pip_codigo)
    ) t 
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
