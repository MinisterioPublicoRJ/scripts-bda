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
            "table_jdbc": "MCPR.MCPR_DOCUMENTO",
            "pk_table_jdbc": "DOCU_DK",
            "update_date_table_jdbc": "DOAA_DT_ALTERACAO",
            "table_hive": "mcpr_documento",
            "fields":
            """
            docu_ano,
            docu_ano_distribuicao,
            docu_atnd_dk,
            docu_cldc_dk,
            docu_comn_dk,
            docu_complemento_protocolo,
            docu_desc_fato_old3,
            docu_dk,
            docu_dt_cadastro,
            docu_dt_cancelamento,
            docu_dt_carga,
            docu_dt_distribuicao,
            docu_dt_entrada_mp,
            docu_dt_fato,
            docu_dt_inicial,
            docu_dt_inicio_prazo_andamento,
            docu_dt_migracao,
            docu_dt_ultima_transmissao,
            docu_fsdc_dk,
            docu_in_anonimo,
            docu_in_apreendido,
            docu_in_atendimento_oficio,
            docu_in_documento_eletronico,
            docu_in_encaminhar_pgj,
            docu_in_horario_fato,
            docu_in_idoso,
            docu_in_preso,
            docu_in_prioridade_infancia,
            docu_in_relatado,
            docu_in_urgente,
            docu_item_dk,
            docu_localizacao_interna,
            docu_mate_dk,
            docu_mdtd_dk,
            docu_motivo_cancelamento,
            docu_nisi_dk,
            docu_nm_parte_protocolo,
            docu_nr_ciac,
            docu_nr_dias_prazo_andamento,
            docu_nr_dias_prazo_documento,
            docu_nr_distribuicao,
            docu_nr_documento_migracao,
            docu_nr_externo,
            docu_nr_folhas,
            docu_nr_mp,
            docu_nr_tj,
            docu_ob_sgp,
            docu_orga_dk_origem,
            docu_orge_orga_dk_deleg_fato,
            docu_orge_orga_dk_deleg_origem,
            docu_orge_orga_dk_vara,
            docu_orgi_orga_dk_carga,
            docu_orgi_orga_dk_entrada,
            docu_orgi_orga_dk_responsavel,
            docu_pcao_dk,
            docu_pced_dk_procedencia,
            docu_pesf_pess_dk_resp_instaur,
            docu_portaria,
            docu_prtl_dk,
            docu_requerente_protocolo,
            docu_tpdc_dk,
            docu_tppc_dk,
            docu_tppr_dk,
            docu_tpst_dk,
            docu_tx_etiqueta,
            docu_vist_dk_aberta,
            docu_vl_causa,
            docu_vl_dano,
            docu_volumes
            """
        },
        {
            "table_jdbc": "MCPR.MCPR_ANDAMENTO",
            "pk_table_jdbc": "PCAO_DK",
            "update_date_table_jdbc": "",
            "partition_column": "pcao_dt_andamento",
            "date_partition_format": 'yyyyMM',
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
            "no_lower_upper_bound": "true",
            "update_date_table_jdbc": "",
            "table_hive": "rh_funcionario"
        },
        {
            "table_jdbc": "MCPR.MCPR_CORRELACIONAMENTO",
            "pk_table_jdbc": "CORR_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mcpr_correlacionamento"
        },
        {
            "table_jdbc": "MPRJ.MPRJ_ORGAO_EXT",
            "pk_table_jdbc": "ORGE_ORGA_DK",
            "update_date_table_jdbc": "",
            "table_hive": "mprj_orgao_ext"
        },
        {
            "table_jdbc": "ORGI.ORGI_ORGAO",
            "pk_table_jdbc": "ORGI_DK",
            "update_date_table_jdbc": "",
            "table_hive": "orgi_orgao"
        },
        {
            "table_jdbc": "GATE.GATE_INFO_TECNICA",
            "pk_table_jdbc": "ITCN_DK",
            "update_date_table_jdbc": "",
            "table_hive": "gate_info_tecnica",
            "fields":
            """
            ITCN_DK,
            ITCN_NR_INFOTECNICA,
            ITCN_DT_CADASTRO,
            ITCN_CDMATRICULA_CADASTRO,
            ITCN_ORGI_DK_SOLICITANTE,
            ITCN_DOCU_DK,
            ITCN_NUGA_DK,
            ITCN_CSPR_DK,
            ITCN_CSSE_DK,
            ITCN_LOCAL,
            ITCN_LOGRADOURO,
            ITCN_NR_LOCAL,
            ITCN_COMPLEMENTO,
            ITCN_BAIRRO,
            ITCN_BAIR_DK,
            ITCN_CIDA_DK,
            ITCN_CEP,
            ITCN_LATITUDE,
            ITCN_LONGITUTE,
            ITCN_COMPL_OPIN_TEC,
            ITCN_IN_DANO_ERARIO,
            ITCN_IN_NIVEL_PROCEDIMENTO,
            ITCN_VAL_DANO_ERARIO,
            ITCN_NUM_CONTRATOS,
            ITCN_IN_TIPO_CONTRATO,
            ITCN_IN_COMP_ANALISE,
            ITCN_IN_ATUALIZACAO_VAL,
            ITCN_NR_INFOTECNICA_VINC,
            ITCN_ATUAL_VAL_DANO_ERARIO,
            ITCN_VAL_INDICIO_DANO,
            ITCN_VAL_COMP_DANO,
            ITCN_DANOS_OUTROS,
            ITCN_NR_SEI_PROTOCOLO
            """
        }
    ]
}