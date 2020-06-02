INSERT INTO {schema_exadata_aux}.atualizacao_pj_pacote SELECT DISTINCT
200 as cod_pct,
"PACOTE AUXILIAR PARA PIPS NO USO DO PROMOTRON" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao
FROM {schema_exadata_aux}.tb_pip_aisp
JOIN {schema_exadata}.orgi_orgao ON orgi_dk = pip_codigo
