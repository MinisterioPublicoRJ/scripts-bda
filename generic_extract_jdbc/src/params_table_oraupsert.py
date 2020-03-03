"""
Observacoes feitas pelos ADS (Oracle):
Retirar as tabelas
-> MCPR.MCPR_DOCUMENTO_ALTERACAO (tabela de uso interno)
-> Todas que comecam AD$ (Auditoria)
-> Todas que terminam com _parcial (sao copias de tabelas)
"""
params = {
    "driver": "oracle.jdbc.driver.OracleDriver",
    "tables": [
        # Schema MCPR
        {
            "table_jdbc": "MCPR.MCPR_HISTORICO_CLASSE_DOC",
            "pk_table_jdbc": "HCDC_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_historico_classe_doc"
        },
        {
            "table_jdbc": "MCPR.MCPR_HISTORICO_FASE_DOC",
            "pk_table_jdbc": "HCFS_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_historico_fase_doc",
            "fields":
            """
            HCFS_DK,HCFS_DOCU_DK,
            HCFS_DT_ENTRADA_MP,HCFS_DT_FIM,
            HCFS_DT_INICIO,HCFS_FASE_ESCOLHIDA,
            HCFS_FSDC_DK,HCFS_IN_REPROCESSAMENTO,
            HCFS_IN_SINCRONIA,HCFS_MATE_DK,HCFS_PCAO_DK,
            HCFS_PCED_DK,HCFS_STAO_DK,HCFS_TPDC_DK,
            HCFS_TPEX_DK,HCFS_TPPR_DK,HCFS_TPST_DK,
            HCFS_TRIGGER_EVENTO,HCFS_USER,HCFS_VIST_DK
            """
        },
        {
            "table_jdbc": "MCPR.MCPR_HISTORICO_MATE_DOCU",
            "pk_table_jdbc": "HMDO_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_historico_mate_docu"
        },
        {
            "table_jdbc": "MCPR.MCPR_HISTORICO_PROCEDENCIA_DOC",
            "pk_table_jdbc": "HPCD_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_historico_procedencia_doc"
        },
        {
            "table_jdbc": "MCPR.MCPR_HISTORICO_RESP_DOCUMENTO",
            "pk_table_jdbc": "HRDC_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_historico_resp_documento"
        },
        {
            "table_jdbc": "MCPR.MCPR_HISTORICO_TIPO_DOC",
            "pk_table_jdbc": "HTPD_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_historico_tipo_doc"
        }
    ]
}