INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote
SELECT
200 as cod_pct,
"PIPs Territoriais 1a CI" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao
FROM (
    SELECT 29934303 AS pip_codigo
    UNION SELECT 29934337
    UNION SELECT 29926583
    UNION SELECT 29926616
    UNION SELECT 29926805
    UNION SELECT 29927047
    UNION SELECT 29933374
    UNION SELECT 29933418
    UNION SELECT 29933469
    UNION SELECT 29933470
    UNION SELECT 29933490
    UNION SELECT 29933502
    UNION SELECT 29933521
    UNION SELECT 29933590
    UNION SELECT 29933830
    UNION SELECT 29933850
    UNION SELECT 29933955
    UNION SELECT 29933967
    UNION SELECT 29933988
    UNION SELECT 29934004
    UNION SELECT 29934012
    UNION SELECT 29934277
    UNION SELECT 29934363
    UNION SELECT 29934376
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
    SELECT 30061694 AS pip_codigo
    UNION SELECT 30061723
    UNION SELECT 30061624
    UNION SELECT 30034384
    UNION SELECT 30061094
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
    SELECT 30069669 AS pip_codigo
    UNION SELECT 30069693
    UNION SELECT 30069732
    UNION SELECT 30070041
    UNION SELECT 30069167
    UNION SELECT 30069433
    UNION SELECT 30069453
    UNION SELECT 30069490
    UNION SELECT 30069516
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
    SELECT 300977 AS pip_codigo
    UNION SELECT 300987
    UNION SELECT 400731
    UNION SELECT 400758
    UNION SELECT 400736
    UNION SELECT 1249804
    UNION SELECT 5679985
    UNION SELECT 6115386
    UNION SELECT 7419262
    UNION SELECT 7419344
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
    SELECT 29941071 AS pip_codigo
    UNION SELECT 29941061
    UNION SELECT 29934401
    UNION SELECT 29934732
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
    SELECT 30034671 AS pip_codigo
    UNION SELECT 30061329
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
    SELECT 29941322 AS pip_codigo
    UNION SELECT 30034664
    UNION SELECT 30069601
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
    SELECT 29941099 AS pip_codigo
    UNION SELECT 29941140
    UNION SELECT 29941222
    UNION SELECT 29941251
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
    SELECT 29941368 AS pip_codigo
    UNION SELECT 30061803
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
    SELECT 30070783 AS pip_codigo
    UNION SELECT 30070806
    ) t
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
