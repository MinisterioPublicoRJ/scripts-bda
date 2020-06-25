with(

    select p.pers_docu_dk,  concat(p.tppe_descricao, ' - ', p.cpfcnpj   , ' - ', p.tppe_cpfcnpj
    from bases.personagem p
) as pers;


--QUERY PARA CONCATENAR COLUNAS DA TABELA PERSONAGEM PARA CADA DOCUMENTO
SELECT d.docu_dk, concat_ws(' | ', collect_set(p.tp_descricao))
FROM bases.documentos d
join (
    select p.pers_docu_dk,  concat(p.tppe_descricao, ' - ', p.cpfcnpj) as tp_descricao
    from bases.personagem p
) p
on p.pers_docu_dk = d.docu_dk
GROUP BY d.docu_dk;