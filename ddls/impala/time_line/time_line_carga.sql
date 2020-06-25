
drop TABLE bases.tl_eventos;

CREATE TABLE bases.tl_eventos
( id string, cpf STRING, dt TIMESTAMP, entidade STRING, cd_entidade STRING, tp_entidade STRING, texto STRING )
partitioned by (tp_evento string)
STORED AS PARQUET;

-- set hive.optimize.index.filter = false;

/*
CREATE TABLE bases.tl_eventos
( id string, tp_evento string, cpf STRING, dt TIMESTAMP, entidade STRING, cd_entidade STRING, tp_entidade STRING, texto STRING )
STORED AS PARQUET;
*/

invalidate metadata;


-- delete from bases.tl_eventos where tp_evento = '+sociedade';
insert into TABLE bases.tl_eventos (id, tp_evento, cpf, dt, entidade, cd_entidade, tp_entidade, texto)
select 
    cast( fnv_hash( concat('+sociedade', nvl(s1.cpf_socio,''), nvl(s1.cnpj,''), nvl(cast(s1.dt_inicio as string),'') )) as string) as id,
    '+sociedade' as tp_evento, 
    s1.cpf_socio as cpf, 
    s1.dt_inicio as dt,
    translate( nvl(pj1.nome,pj1.nome_fantasia) ,'"','') as entidade,
    s1.cnpj as cd_entidade,
    'PJ' as tp_entidade,
    concat('Nome Fantasia: ', translate(pj1.nome_fantasia,'"',pj1.nome), '|', group_concat( concat( cast( s2.percentual * 100 as string) , '% - ' , s2.tipo_socio , ' - ' ,pf2.nome), '|')) as texto

from 
    bases.lc_socio s1 
    left join bases.lc_cnpj pj1 on s1.cnpj = pj1.num_cnpj
    left join bases.lc_socio s2 on s1.cnpj = s2.cnpj and s2.dt_inicio <= s1.dt_inicio 
    left join bases.pessoa_fisica pf2 on s2.cpf_socio = pf2.num_cpf

-- where s1.cpf_socio = '11452244740'
    
group by
    s1.cpf_socio, 
    s1.dt_inicio,
    pj1.nome_fantasia,
    pj1.nome,
    s1.cnpj

;

--delete from bases.tl_eventos where tp_evento = '-sociedade';
insert into TABLE bases.tl_eventos (id, tp_evento, cpf, dt, entidade, cd_entidade, tp_entidade, texto)
select 
    cast( fnv_hash( concat('-sociedade', nvl(s1.cpf_socio,''), nvl(s1.cnpj,''), nvl(cast(s1.dt_fim as string),'') )) as string) as id,    
    '-sociedade' as tp_evento, 
    s1.cpf_socio as cpf, 
    s1.dt_fim as dt,
    translate( nvl(pj1.nome,pj1.nome_fantasia) ,'"','') as entidade,
    s1.cnpj as cd_entidade,
    'PJ' as tp_entidade,    
    concat('Nome Fantasia: ', translate(pj1.nome_fantasia,'"',pj1.nome),'|Tempo de Sociedade: ',cast(cast(months_between(v1.dt_fim, v1.dt_inicio) as integer) as string),' meses ','|', group_concat( concat( cast( s2.percentual * 100 as string) , '% - ' , s2.tipo_socio , ' - ' ,pf2.nome), '|')) as texto
    

from 
    bases.lc_socio s1 
    left join bases.lc_cnpj pj1 on s1.cnpj = pj1.num_cnpj
    left join bases.lc_socio s2 on s1.cnpj = s2.cnpj and (s2.dt_fim > s1.dt_fim or s2.dt_fim is null) and  s2.dt_inicio <= s1.dt_fim
    left join bases.pessoa_fisica pf2 on s2.cpf_socio = pf2.num_cpf

-- where s1.cpf_socio = '11452244740'
where s1.dt_fim is not null --and s1.cpf_socio = '11452244740'
    
group by
    s1.cpf_socio, 
    s1.dt_fim,
    pj1.nome_fantasia,
    pj1.nome,
    s1.cnpj

;

--delete from bases.tl_eventos where tp_evento = '+emprego';
insert into TABLE bases.tl_eventos (id, tp_evento, cpf, dt, entidade, cd_entidade, tp_entidade, texto)
select 
    cast( fnv_hash( concat('+emprego', nvl(v1.cpf,''), nvl(v1.cnpj,''), nvl(cast(v1.dt_inicio as string),''))) as string) as id,
    '+emprego' as tp_evento, 
    v1.cpf as cpf, 
    v1.dt_inicio as dt,
    translate( nvl(pj1.nome,pj1.nome_fantasia) ,'"','') as entidade,
    v1.cnpj as cd_entidade,
    'PJ' as tp_entidade,
    tp_vinculo as texto
    --'' as texto
    --concat(v1.tp_vinculo  ,'|', cast(count(v2.cpf) as varchar) , ' outros funcionários')  as texto

from 
    bases.lc_vinculo_trabalhista v1 
    left join bases.lc_cnpj pj1 on cast(v1.cnpj as bigint) = cast(pj1.num_cnpj as bigint)
    --left join bases.lc_vinculo_trabalhista v2 on v1.cnpj = v2.cnpj and v2.dt_inicio <= v1.dt_inicio 

-- where v1.cpf = '11452244740'
/*
group by 
    v1.cpf, 
    v1.dt_inicio,
    pj1.nome_fantasia,
    v1.cnpj,
    v1.tp_vinculo
  */  
;

--delete from bases.tl_eventos where tp_evento = '-emprego';
insert into TABLE bases.tl_eventos (id, tp_evento, cpf, dt, entidade, cd_entidade, tp_entidade, texto)
select 
    cast( fnv_hash( concat('-emprego', nvl(v1.cpf,''), nvl(v1.cnpj,''), nvl(cast(v1.dt_fim as string),''))) as string) as id,
    '-emprego' as tp_evento, 
    v1.cpf as cpf, 
    v1.dt_fim as dt,
    translate( nvl(pj1.nome,pj1.nome_fantasia) ,'"','') as entidade,
    v1.cnpj as cd_entidade,
    'PJ' as tp_entidade,
    concat('Tempo de Vínculo: ', cast(cast(months_between(v1.dt_fim, v1.dt_inicio) as integer) as string),' meses ', ' | ',v1.tp_vinculo) as texto
    --'' as texto
    --concat(v1.tp_vinculo  ,'|', cast(count(v2.cpf) as varchar) , ' outros funcionários')  as texto

from 
    bases.lc_vinculo_trabalhista v1 
    left join bases.lc_cnpj pj1 on cast(v1.cnpj as bigint) = cast(pj1.num_cnpj as bigint)
    --left join bases.lc_vinculo_trabalhista v2 on v1.cnpj = v2.cnpj and v2.dt_inicio <= v1.dt_fim 

where v1.dt_fim is not null --and  v1.cpf = '11452244740'
/*
group by 
    v1.cpf, 
    v1.dt_fim,
    pj1.nome_fantasia,
    pj1.nome,
    v1.cnpj,
    v1.tp_vinculo
*/    
;

--delete from bases.tl_eventos where tp_evento = '+paternidade';
insert into TABLE bases.tl_eventos (id, tp_evento, cpf, dt, entidade, cd_entidade, tp_entidade, texto)
select
    cast( fnv_hash( concat('+paternidade', nvl(p.num_cpf,''), nvl(f.num_cpf,''))) as string) as id,
    '+paternidade' as tp_evento,
    p.num_cpf as cpf,
    f.data_nascimento as dt,
    translate(f.nome,'"','') as entidade,
    f.num_cpf as cd_entidade,
    'PF' as tp_entidade,
    concat('Mãe: ', f.nome_mae) as texto
from dadossinapse.pessoa_pai_ope v
join dadossinapse.pessoa_fisica_opv f on v.start_node = f.uuid
join dadossinapse.pessoa_fisica_opv p on v.end_node = p.uuid
--where p.num_cpf = '11452244740'


;

--delete from bases.tl_eventos where tp_evento = '+maternidade';
insert into TABLE bases.tl_eventos (id, tp_evento, cpf, dt, entidade, cd_entidade, tp_entidade, texto)
select 
    cast( fnv_hash( concat('+maternidade', nvl(m.num_cpf,''), nvl(f.num_cpf,''))) as string) as id,
    '+maternidade' as tp_evento,
    m.num_cpf as cpf,
    f.data_nascimento as dt,
    translate(f.nome,'"','') as entidade,
    f.num_cpf as cd_entidade,
    'PF' as tp_entidade,
    concat('Pai: ', f.nome_pai) as texto
from dadossinapse.pessoa_mae_ope v
join dadossinapse.pessoa_fisica_opv f on v.start_node = f.uuid
join dadossinapse.pessoa_fisica_opv m on v.end_node = m.uuid
--where m.num_cpf = '10069222703'
;
/*
--delete from bases.tl_eventos where tp_evento = '+multa';
insert into TABLE bases.tl_eventos (id, tp_evento, cpf, dt, entidade, cd_entidade, tp_entidade, texto)
select
    cast( fnv_hash( concat('+multa', nvl(ident2,''), nvl(autoinfra,''))) as string) as id,
    '+multa' as tp_evento,
    substr(ident2,-11) as cpf,
    dt_inf as dt,
    translate(tve_tab_descricao_marca,'"','') as entidade,
    pl_vei_inf as cd_entidade,
    'VEI' as tp_entidade,
    concat('Placa: ', pl_vei_inf, '|Infração: ', ds_inf_tab, '|Local: ',  localinfra) as texto

from bases.detran_multa
;
*/

--delete from bases.tl_eventos where tp_evento = '+mgp';
insert into TABLE bases.tl_eventos (id, tp_evento, cpf, dt, entidade, cd_entidade, tp_entidade, texto)
select  
    cast( fnv_hash( concat('+mgp', nvl(p.cpfcnpj,''), nvl(d.nr_mp,''), nvl(p.tppe_descricao,''))) as string) as id,
    '+mgp' as tp_evento,
    p.cpfcnpj as cpf,
    d.dt_cadastro as dt,
    d.nr_mp as entidade,
    d.nr_mp as cd_entidade,
    'MGP' as tp_entidade,
    concat( replace(d.cldc_ds_hierarquia,'|','-') , '|Personagens:|',  group_concat( concat( p2.tppe_descricao , ' - ' , p2.pess_nm_pessoa), '|') ) as texto

from dadossinapse.personagem_opv p
left join dadossinapse.personagem_opv p2 on p.pers_docu_dk = p2.pers_docu_dk    
left join dadossinapse.documento_opv d on p.pers_docu_dk = d.docu_dk

where length(p.cpfcnpj) = 11 and p.cpfcnpj = '11452244740'

group by 
    p.tppe_descricao,
    p.cpfcnpj,
    d.dt_cadastro,
    d.nr_mp,
    d.cldc_ds_hierarquia
;