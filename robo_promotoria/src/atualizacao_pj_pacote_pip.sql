INSERT INTO exadata_aux_dev.atualizacao_pj_pacote SELECT DISTINCT
200 as cod_pct,
"PACOTE AUXILIAR PARA PIPS NO USO DO PROMOTRON" as pacote_atribuicao,
pip_codigo as id_orgao,
orgi_nm_orgao_abrev as orgao_codamp,
orgi_nm_orgao as orgi_nm_orgao
FROM exadata_aux_dev.tb_pip_aisp
JOIN exadata_dev.orgi_orgao ON orgi_dk = pip_codigo;
