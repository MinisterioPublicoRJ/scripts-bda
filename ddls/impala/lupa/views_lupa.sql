drop view lupa.lp_craai_promotorias;
create view lupa.lp_craai_promotorias as
SELECT 
    cast(orlw_regi_dk as string) as orlw_regi_dk,
    orgao.ORGI_CDORGAO,
    orgao.ORGI_NM_ORGAO,
    craai.ORLW_REGI_NM_REGIAO,
    tipoorgao.TPOR_DK,
    tipoorgao.TPOR_DS_TP_ORGAO,
    'PRO' destino
FROM 
    exadata.orgi_orgao orgao
    INNER JOIN exadata.ORGI_VW_ORGAO_LOCAL_ATUAL craai ON craai.ORLW_ORGI_CDORGAO = orgao.ORGI_CDORGAO
    INNER JOIN exadata.ORGI_TP_ORGAO tipoorgao ON tipoorgao.tpor_dk = orgao.ORGI_TPOR_DK
WHERE 
    orgi_tpor_dk = 1 
    AND (
        (orgi_dt_fim is NULL) 
        OR orgi_dt_fim >= now()
    )
order by orgao.orgi_nm_orgao
limit 100000000;

drop view lupa.lp_predio_promotorias;
create view lupa.lp_predio_promotorias as
SELECT 
    predio.codigo_do_imovel as cod_imovel,
    orgao.ORGI_CDORGAO,
    orgao.ORGI_NM_ORGAO,
    'PRO' destino
FROM 
    exadata.orgi_orgao orgao
    INNER JOIN (
        SELECT codigo_do_imovel, codigo_cc
        FROM opengeo.financeiro
        GROUP BY codigo_do_imovel, codigo_cc
    ) as predio ON predio.codigo_cc = cast(orgao.ORGI_DK as string)
WHERE 
    orgi_tpor_dk = 1
    AND (
        (orgi_dt_fim is NULL) 
        OR orgi_dt_fim >= now()
    )
order by orgao.orgi_nm_orgao
limit 100000000;

------------------------------

drop view lupa.lp_craai_outros_orgaos;
create view lupa.lp_craai_outros_orgaos as
SELECT 
    cast(orlw_regi_dk as string) as orlw_regi_dk,
    orgao.ORGI_CDORGAO,
    orgao.ORGI_NM_ORGAO,
    craai.ORLW_REGI_NM_REGIAO,
    tipoorgao.TPOR_DK,
    tipoorgao.TPOR_DS_TP_ORGAO,
    'ORG' destino
FROM exadata.orgi_orgao orgao
INNER JOIN exadata.ORGI_VW_ORGAO_LOCAL_ATUAL craai ON
	craai.ORLW_ORGI_CDORGAO = orgao.ORGI_CDORGAO
INNER JOIN exadata.ORGI_TP_ORGAO tipoorgao ON
	tipoorgao.tpor_dk = orgao.ORGI_TPOR_DK
WHERE orgi_tpor_dk not in (1,3,5) AND (
	(orgi_dt_fim is NULL) or orgi_dt_fim >= now()
)
order by orgao.orgi_nm_orgao
limit 100000000;

------------------------------

drop view lupa.lp_predio_outros_orgaos;
create view lupa.lp_predio_outros_orgaos as
SELECT 
    predio.codigo_do_imovel as cod_imovel,
    orgao.ORGI_CDORGAO,
    orgao.ORGI_NM_ORGAO,
    'ORG' destino
FROM exadata.orgi_orgao orgao
INNER JOIN (
    SELECT codigo_do_imovel, codigo_cc
    FROM opengeo.financeiro
    GROUP BY codigo_do_imovel, codigo_cc
) as predio ON predio.codigo_cc = cast(orgao.ORGI_DK as string)
WHERE orgi_tpor_dk not in (1,3, 5) AND (
	(orgi_dt_fim is NULL) or orgi_dt_fim >= now()
)
order by orgao.orgi_nm_orgao
limit 100000000;

------------------------------------

drop view lupa.lp_orgao_acervo;
create view lupa.lp_orgao_acervo as 
select cast(cdorgao as string) cdorgao, dt, cast(entradas-saidas as int) acumulo, tipo
from cluster.painel_acervo
where dt > add_months(now(),-8);

------------------------------------------
drop view lupa.lp_craai_procuradorias;
create view lupa.lp_craai_procuradorias as
SELECT
    cast(orlw_regi_dk as string) orlw_regi_dk,
    orgao.ORGI_CDORGAO,
    orgao.ORGI_NM_ORGAO,
    craai.ORLW_REGI_NM_REGIAO,
    tipoorgao.TPOR_DK,
    tipoorgao.TPOR_DS_TP_ORGAO,
    'PRC' destino
FROM exadata.orgi_orgao orgao
INNER JOIN exadata.ORGI_VW_ORGAO_LOCAL_ATUAL craai ON
	craai.ORLW_ORGI_CDORGAO = orgao.ORGI_CDORGAO
INNER JOIN exadata.ORGI_TP_ORGAO tipoorgao ON
	tipoorgao.tpor_dk = orgao.ORGI_TPOR_DK
WHERE orgi_tpor_dk = 3 AND (
	(orgi_dt_fim is NULL) or orgi_dt_fim >= now()
)
ORDER BY orgao.orgi_nm_orgao
LIMIT 10000;

---------------------------------------------------------------------

drop view lupa.lp_predio_procuradorias;
create view lupa.lp_predio_procuradorias as
SELECT
    predio.codigo_do_imovel as cod_imovel,
    orgao.ORGI_CDORGAO,
    orgao.ORGI_NM_ORGAO,
    'PRC' destino
FROM exadata.orgi_orgao orgao
INNER JOIN (
    SELECT codigo_do_imovel, codigo_cc
    FROM opengeo.financeiro
    GROUP BY codigo_do_imovel, codigo_cc
) as predio ON predio.codigo_cc = cast(orgao.ORGI_DK as string)
WHERE orgi_tpor_dk = 3 AND (
	(orgi_dt_fim is NULL) or orgi_dt_fim >= now()
)
ORDER BY orgao.orgi_nm_orgao
LIMIT 10000;

---------------------------------------------------------------------

drop view lupa.lp_promotores;
create view lupa.lp_promotores as
select distinct
    mmpm_cdorgao,
    mmpm_nome,
    concat(
        mmpm_cargo, '@',
        'Concurso: ', mmpm_romano, ' ano ', cast(cast(mmpm_anoconcurso as int) as string), '@',
        'Nascimento: ', mmpm_dtnasc, '@',
        'Atribuição: ', isnull(mmpm_pgj_funcao, "--")
    ) detalhes,
    base64_vtfu_foto foto
from exadata.mmps_adm_rh_mov_prom
inner join exadata.mmps_vw_rh_foto_func vw_foto on
    mmps_adm_rh_mov_prom.mmpm_matricula = vw_foto.vtfu_cdmatricula


order by cdorgao, dt asc
limit 1000000;

-------------------------------------------------

drop view lupa.lp_promotores;
create view lupa.lp_promotores as
select distinct
    mmpm_cdorgao,
    mmpm_nome,
    concat(
        mmpm_cargo, '@',
        'Concurso: ', mmpm_romano, ' ano ', cast(cast(mmpm_anoconcurso as int) as string), '@',
        'Nascimento: ', mmpm_dtnasc, '@',
        'Atribuição: ', isnull(mmpm_pgj_funcao, "--")
    ) detalhes,
    base64_vtfu_foto foto
from exadata.mmps_adm_rh_mov_prom
inner join exadata.mmps_vw_rh_foto_func vw_foto on
    mmps_adm_rh_mov_prom.mmpm_matricula = vw_foto.vtfu_cdmatricula;


-------------------------------------------------------------

drop view lupa.lp_servidores ;
create view lupa.lp_servidores as
select 
    lotacao.cdorgao_lot_atual cdorgao,
    lotacao.nome mmsv_nome,
    concat(
        lotacao.dstipfunc, '@',
        'Nascimento: ', mmsv_dtnasc, '@',
        'Função: ', lotacao.cargo_ou_cc_ou_funcao, '@',
        'Custo Total MPRJ: R$ ', cast(cast(col_k_custo_total_mprj as decimal(9,2)) as string)
    ) detalhes,
    base64_vtfu_foto foto
from exadata.RH_VW_RELAT_CUSTOS_MP_MAPAS lotacao
inner join exadata.mmps_adm_rh_mov_serv servidor on 
    servidor.mmsv_matricula = lotacao.cdmatricula
inner join exadata.mmps_vw_rh_foto_func vw_foto on
    servidor.mmsv_matricula = vw_foto.vtfu_cdmatricula;
    



------------------------------------------

drop view lupa.lp_orgaos_auxiliares;
create view lupa.lp_orgaos_auxiliares as 
select 
    orgao.orgi_nm_orgao,
    cast(orgao.orgi_dk as string) orgao_auxilizar,
    cast(auxilia.orau_orgi_dk_auxilia as string) orgao_auxiliado
from exadata.orgi_auxilia auxilia
inner join exadata.orgi_orgao orgao on
    auxilia.orau_orgi_dk = orgao.orgi_dk
where orgao.orgi_tpor_dk not in (5);

----------------------------------------------------


drop view lupa.lp_predio_custos_por_orgao;
create view lupa.lp_predio_custos_por_orgao as
select codigo_do_imovel, centro_de_custos, sum(total) total
from opengeo.financeiro
where extract(year from competencia) >= extract(year from now())
group by codigo_do_imovel, centro_de_custos
order by total desc
limit 1000000000;

----------------------------------------------------

drop view lupa.lp_predio_custos_tipo;
create view lupa.lp_predio_custos_tipo as
select codigo_do_imovel, tipo_de_custo, sum(total) total
from opengeo.financeiro
where extract(year from competencia) >= extract(year from now())
group by codigo_do_imovel, tipo_de_custo
order by total desc
limit 10000000;



---------------------------------------------

drop view lupa.lp_procuradores;
create view lupa.lp_procuradores as
select distinct
    mmpc_cdorgao,
    mmpc_nome,
    concat(
        mmpc_cargo, '@',
        'Concurso: ', mmpc_concurso_romano, ' ano ', cast(cast(mmpc_anoconcurso as int) as string), '@',
        'Nascimento: ', mmpc_dtnasc, '@',
        'Atribuição: ', isnull(mmpc_pgj, "--")
    ) detalhes,
    base64_vtfu_foto foto
from exadata.mmps_adm_rh_mov_proc
inner join exadata.mmps_vw_rh_foto_func vw_foto on
    mmps_adm_rh_mov_proc.mmpc_matricula = vw_foto.vtfu_cdmatricula
    
    
    
------------------------------------------------------------------------------------------------


drop view lupa.lp_prefeitos;
create view lupa.lp_prefeitos as 
SELECT 
       Cast(cod_municipio AS STRING) cod_municipio,
       cod_craai,
       governantes.nome,
       Concat(
       Cast(ano_eleicao AS STRING), '@',
       '<a href="', url_tse, '" target="blank">Perfil no Tribunal Superior Eleitoral</a>@',
       'Site: <a href="http://', trim(site), '" target="blank">', site, '</a>@',
       'Telefones: ',
       telefone, '@', 'Endereço: ', endereco) detalhes,
       governantes.base64_foto foto,
       municipio
FROM   opengeo.lupa_prefeituras prefeituras
       INNER JOIN opengeo.lupa_governantes_rj governantes
               ON governantes.cod_ibge = prefeituras.cod_municipio  ;
    
--------------------------------------------------------------------------------



drop view lupa.lp_dados_municipio;
create view lupa.lp_dados_municipio as 
SELECT 
       Cast(prefeituras.cod_municipio AS STRING) cod_municipio,
       cod_craai,
       '' as nome,
       Concat(
         'Data de Fundação: ', municipio_flag.aniversario, '@',
         'Legislação de Criação: ', municipio_flag.criacao, '@',
         'Gentílico: ', prefeituras.gentilico
       ) detalhes,
       flag as foto
FROM   opengeo.lupa_prefeituras prefeituras
inner join opengeo.municipio_flag on
    municipio_flag.cod_municipio = prefeituras.cod_municipio
--------------------------------------------------------------------------------


drop view lupa.lp_governantes;
create view lupa.lp_governantes as 
select
    cast(cod_ibge as string) cod_ibge,
    nome,
    concat('Mandato: ', cast(ano_eleicao+1 as string), '/', cast(ano_eleicao +4 as string)) detalhes,
    base64_foto foto
from opengeo.lupa_governantes_rj governantes;


---------------------------------------------------------------------------------


drop view lupa.lp_movimentacao_rh;
create view lupa.lp_movimentacao_rh as 
SELECT  rh_f.cdmatricula,
        rh_f.cdcargo, 
        rh_c.nmcargo,
        rh_f.cdtipfunc, 
        rh_tf.dstipfunc,
        year(current_timestamp()) - year(rh_f.dtnasc) as idade, 
        rh_f.sexo,
        org.codigo_municipio,
        org.municipio
FROM exadata.rh_funcionario rh_f
join opengeo.lupa_orgaos_mprj  org
on cast(rh_f.cdorgao as int) = org.codigo_mgo
join exadata.rh_cargos rh_c
on rh_c.cdcargo = rh_f.cdcargo
left join exadata_aux.tipo_funcionario rh_tf
on rh_tf.cdtipfunc = cast(rh_f.cdtipfunc as int)
