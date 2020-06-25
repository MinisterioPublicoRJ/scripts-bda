INSERT OVERWRITE TABLE exadata_aux.tb_acervo 
PARTITION (dt_partition)
SELECT  cod_orgao,
        cod_atribuicao,
        acervo,
        tipo_acervo,
        dt_inclusao,
        dt_partition
FROM exadata_aux_dev.tb_acervo
---EOS---

INSERT OVERWRITE TABLE exadata_aux.tb_distribuicao
PARTITION (dt_partition)
SELECT  cod_atribuicao,
        minimo,
        maximo,
        media,
        primeiro_quartil,
        mediana,
        terceiro_quartil,
        iqr,
        lout,
        hout,
        dt_inclusao,
        dt_partition
FROM exadata_aux_dev.tb_distribuicao
---EOS---

CREATE TABLE exadata_aux.atualizacao_pj_pacote as
SELECT  cod_pct,
        pacote_atribuicao,
        id_orgao,
        orgao_codamp,
        orgi_nm_orgao
FROM exadata_aux_dev.atualizacao_pj_pacote