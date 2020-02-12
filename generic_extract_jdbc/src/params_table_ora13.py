"""
Observacoes feitas pelos ADS (Oracle):
Retirar as tabelas
-> MCPR.MCPR_DOCUMENTO_ALTERACAO (tabela de uso interno)
-> Todas que comecam AD$ (Auditoria)
-> Todas que terminam com _parcial (sao copias de tabelas)
"""
params = {
    "driver": "oracle.jdbc.driver.OracleDriver",
    "schema_hdfs": "exadata_dev",
    "tables": [
        # Schema MCPR
        {
            "table_jdbc": "MCPR.MCPR_DOCUMENTO",
            "pk_table_jdbc": "DOCU_DK",
            "update_date_table_jdbc": "DOAA_DT_ALTERACAO",
            "table_hive": "mcpr_documento",
            "fields":
            """
            DOCU_ANO,
            DOCU_ANO_DISTRIBUICAO,
            DOCU_ATND_DK,
            DOCU_CLDC_DK,
            DOCU_COMN_DK,
            DOCU_COMPLEMENTO_PROTOCOLO,
            DOCU_DESC_FATO_OLD3,
            DOCU_DK,
            DOCU_DT_CADASTRO,
            DOCU_DT_CANCELAMENTO,
            DOCU_DT_CARGA,
            DOCU_DT_DISTRIBUICAO,
            DOCU_DT_ENTRADA_MP,
            DOCU_DT_FATO,
            DOCU_DT_INICIAL,
            DOCU_DT_INICIO_PRAZO_ANDAMENTO,
            DOCU_DT_MIGRACAO,
            DOCU_DT_ULTIMA_TRANSMISSAO,
            DOCU_FSDC_DK,
            DOCU_IN_ANONIMO,
            DOCU_IN_APREENDIDO,
            DOCU_IN_ATENDIMENTO_OFICIO,
            DOCU_IN_DOCUMENTO_ELETRONICO,
            DOCU_IN_ENCAMINHAR_PGJ,
            DOCU_IN_HORARIO_FATO,
            DOCU_IN_IDOSO,
            DOCU_IN_PRESO,
            DOCU_IN_PRIORIDADE_INFANCIA,
            DOCU_IN_RELATADO,
            DOCU_IN_URGENTE,
            DOCU_ITEM_DK,
            DOCU_LOCALIZACAO_INTERNA,
            DOCU_MATE_DK,
            DOCU_MDTD_DK,
            DOCU_MOTIVO_CANCELAMENTO,
            DOCU_NISI_DK,
            DOCU_NM_PARTE_PROTOCOLO,
            DOCU_NR_CIAC,
            DOCU_NR_DIAS_PRAZO_ANDAMENTO,
            DOCU_NR_DIAS_PRAZO_DOCUMENTO,
            DOCU_NR_DISTRIBUICAO,
            DOCU_NR_DOCUMENTO_MIGRACAO,
            DOCU_NR_EXTERNO,
            DOCU_NR_FOLHAS,
            DOCU_NR_MP,
            DOCU_NR_TJ,
            DOCU_OB_SGP,
            DOCU_ORGA_DK_ORIGEM,
            DOCU_ORGE_ORGA_DK_DELEG_FATO,
            DOCU_ORGE_ORGA_DK_DELEG_ORIGEM,
            DOCU_ORGE_ORGA_DK_VARA,
            DOCU_ORGI_ORGA_DK_CARGA,
            DOCU_ORGI_ORGA_DK_ENTRADA,
            DOCU_ORGI_ORGA_DK_RESPONSAVEL,
            DOCU_PCAO_DK,
            DOCU_PCED_DK_PROCEDENCIA,
            DOCU_PESF_PESS_DK_RESP_INSTAUR,
            DOCU_PORTARIA,
            DOCU_PRTL_DK,
            DOCU_REQUERENTE_PROTOCOLO,
            DOCU_TPDC_DK,
            DOCU_TPPC_DK,
            DOCU_TPPR_DK,
            DOCU_TPST_DK,
            DOCU_TX_ETIQUETA,
            DOCU_VIST_DK_ABERTA,
            DOCU_VL_CAUSA,
            DOCU_VL_DANO,
            DOCU_VOLUMES
            """
        },
        {
            "table_jdbc": "MCPR.MCPR_ANDAMENTO",
            "pk_table_jdbc": "PCAO_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_andamento",
            "fields":
            """
            PCAO_CDMATRICULA_CANCELOU,PCAO_CDMATRICULA_REG_ANDAMENTO,
            PCAO_CPF_CANCELOU_ANDAMENTO,PCAO_CPF_REG_ANDAMENTO,
            PCAO_DK,PCAO_DT_ANDAMENTO,PCAO_DT_CANCELAMENTO,
            PCAO_DT_REGISTRO,PCAO_FSDC_DK_ANDAMENTO,
            PCAO_IN_ANDAMENTO_SEM_CARGA,PCAO_IN_SIGILO_ANDAMENTO,
            PCAO_JUSTIFICATIVA_SIGILO_DOCU,PCAO_MOTIVO_CANCELAMENTO,
            PCAO_OBSERVACAO,PCAO_ORGI_ORGA_DK_DEST_TRAMIT,
            PCAO_TMPL_DK_USADO,PCAO_TPSA_DK,
            PCAO_TPST_DK_ANTERIOR,PCAO_VIST_DK
            """
        },
        {
            "table_jdbc": "MCPR.MCPR_ARQUIVAMENTO",
            "pk_table_jdbc": "MARQ_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_arquivamento"
        },
        {
            "table_jdbc": "MCPR.MCPR_ITEM_MOVIMENTACAO",
            "pk_table_jdbc": "ITEM_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_item_movimentacao"
        },
        {
            "table_jdbc": "MCPR.MCPR_ITEM_MOVIMENTACAO_PARCIAL",
            "pk_table_jdbc": "ITEM_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_item_movimentacao_parcial"
        },
        {
            "table_jdbc": "MCPR.MCPR_MOVIMENTACAO",
            "pk_table_jdbc": "MOVI_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_movimentacao"
        },
        {
            "table_jdbc": "MCPR.MCPR_PERSONAGEM",
            "pk_table_jdbc": "PERS_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_personagem"
        },
        {
            "table_jdbc": "MCPR.MCPR_PERSONAGEM_ANDAMENTO",
            "pk_table_jdbc": "PEAN_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_personagem_andamento"
        },
        {
            "table_jdbc": "MCPR.MCPR_PESSOA",
            "pk_table_jdbc": "PESS_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_pessoa"
        },
        {
            "table_jdbc": "MCPR.MCPR_PESSOA_FISICA",
            "pk_table_jdbc": "PESF_PESS_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_pessoa_fisica"
        },
        {
            "table_jdbc": "MCPR.MCPR_PESSOA_JURIDICA",
            "pk_table_jdbc": "PESJ_PESS_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_pessoa_juridica",
            "fields":
            """
            PESJ_APLICACAO_ATUALIZOU,PESJ_CNPJ,
            PESJ_DT_ALVARA_FUNCIONAMENTO,
            PESJ_DT_FUND,PESJ_DT_REGIST_CONTRATO_SOCIAL,
            PESJ_DT_ULTIMA_ATUALIZACAO,PESJ_DT_VALID_CERT_APROV_BOMB,
            PESJ_INSCR_ESTADUAL,PESJ_INSCR_MUNICIPAL,
            PESJ_NM_PESSOA_JURIDICA,PESJ_NM_RESPONSAVEL,
            PESJ_NOME_FANTASIA,PESJ_NR_ALVARA_FUNCIONAMENTO,
            PESJ_NR_CERTIF_APROV_BOMBEIRO,PESJ_PESS_DK,
            PESJ_SIGLA,PESJ_TPOE_DK
            """
        },
        {
            "table_jdbc": "MCPR.MCPR_SUB_ANDAMENTO",
            "pk_table_jdbc": "STAO_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_sub_andamento"
        },
        {
            "table_jdbc": "MCPR.MCPR_VISTA",
            "pk_table_jdbc": "VIST_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_vista",
            "fields":
            """
            VIST_DK,VIST_DOCU_DK,
            VIST_DT_ABERTURA_VISTA,
            VIST_DT_EMISSAO_FOLHA_VISTA,
            VIST_DT_FECHAMENTO_VISTA,
            VIST_DT_REGISTRO,VIST_FINA_DK,
            VIST_IN_CONTABILIZA_CGMP,VIST_ORGI_ORGA_DK,
            VIST_PCED_DK,VIST_PESF_PESS_DK_RESP_ANDAM,
            VIST_TMPL_DK_USADO
            """
        },
        {
            "table_jdbc": "RH.FUNCIONARIO",
            "pk_table_jdbc": "CDMATRICULA",
            "update_date_table_jdbc": "",
            "table_hive": "rh_funcionario"
        }
    ]
}