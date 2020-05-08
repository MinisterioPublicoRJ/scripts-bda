CREATE VIEW IF NOT EXISTS detran.vw_regcivil AS
    SELECT orig.*
    FROM detran.tb_regcivil orig
    INNER JOIN
    (SELECT nu_rg, MAX(dt_expedicao_carteira) ultima_expedicao
       FROM detran.tb_regcivil GROUP BY nu_rg) sub
       ON orig.nu_rg = sub.nu_rg
       AND orig.dt_expedicao_carteira = sub.ultima_expedicao
    ORDER BY year DESC;
