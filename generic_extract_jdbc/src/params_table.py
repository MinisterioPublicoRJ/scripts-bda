"""
Observacoes feitas pelos ADS (Oracle):
Retirar as tabelas
-> MCPR.MCPR_DOCUMENTO_ALTERACAO (tabela de uso interno)
-> Todas que comecam AD$ (Auditoria)
-> Todas que terminam com _parcial (sao copias de tabelas)
"""
params = {
    "driver": "oracle.jdbc.driver.OracleDriver",
    "schema_hdfs": "exadata",
    "tables": [
        # Schema MCPR
        {
            "table_oracle": "MCPR.MCPR_ACOMPANHAMENTO_DOCUMENTO",
            "pk_table_oracle": "ACOM_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_acompanhamento_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_ACOMPANHAMENTO_REGRA",
            "pk_table_oracle": "ACRE_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_acompanhamento_regra"
        },
        {
            "table_oracle": "MCPR.MCPR_ADVOGADO",
            "pk_table_oracle": "ADGO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_advogado"
        },
        {
            "table_oracle": "MCPR.MCPR_ADVOGADO_PERSONAGEM",
            "pk_table_oracle": "ADPR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_advogado_personagem"
        },
        {
            "table_oracle": "MCPR.MCPR_AMBIENTE_AGRESSAO",
            "pk_table_oracle": "AMAG_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_ambiente_agressao"
        },
        {
            "table_oracle": "MCPR.MCPR_ANDAMENTO_FASE_EXCECAO",
            "pk_table_oracle": "TPEX_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_andamento_fase_excecao"
        },
        {
            "table_oracle": "MCPR.MCPR_ANDAMENTO_OBRIGATORIO",
            "pk_table_oracle": "ANOB_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_andamento_obrigatorio"
        },
        {
            "table_oracle": "MCPR.MCPR_ANEXO_PRESTACAO_CONTA",
            "pk_table_oracle": "MAPC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_anexo_prestacao_conta"
        },
        {
            "table_oracle": "MCPR.MCPR_ANOTACAO_DOCUMENTO",
            "pk_table_oracle": "ANTD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_anotacao_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_ANUARIO",
            "pk_table_oracle": "ANUA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_anuario"
        },
        {
            "table_oracle": "MCPR.MCPR_ARQUIVAMENTO",
            "pk_table_oracle": "MARQ_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_arquivamento"
        },
        {
            "table_oracle": "MCPR.MCPR_ARTIGO",
            "pk_table_oracle": "ARTG_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_artigo"
        },
        {
            "table_oracle": "MCPR.MCPR_ASSUNTO",
            "pk_table_oracle": "ASSU_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_assunto"
        },
        {
            "table_oracle": "MCPR.MCPR_ASSUNTO_ATENDIMENTO",
            "pk_table_oracle": "ASAT_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_assunto_atendimento"
        },
        {
            "table_oracle": "MCPR.MCPR_ASSUNTO_DOCTO_CONTR_INTEG",
            "pk_table_oracle": "ARLT_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_assunto_docto_contr_integ"
        },
        {
            "table_oracle": "MCPR.MCPR_ASSUNTO_DOCUMENTO",
            "pk_table_oracle": "ASDO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_assunto_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_ASSUNTOS_DE_ATENDIMENTO",
            "pk_table_oracle": "ADAT_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_assuntos_de_atendimento"
        },
        {
            "table_oracle": "MCPR.MCPR_AVISO_APLICACAO",
            "pk_table_oracle": "AISO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_aviso_aplicacao"
        },
        {
            "table_oracle": "MCPR.MCPR_BEM_IMOVEL",
            "pk_table_oracle": "BMMV_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_bem_imovel"
        },
        {
            "table_oracle": "MCPR.MCPR_BEM_MOBILIARIO",
            "pk_table_oracle": "BMMB_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_bem_mobiliario"
        },
        {
            "table_oracle": "MCPR.MCPR_CAIXA_ARQUIVO_VAZIA",
            "pk_table_oracle": "MCAV_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_caixa_arquivo_vazia"
        },
        {
            "table_oracle": "MCPR.MCPR_CLASSE_DOCTO_MP",
            "pk_table_oracle": "CLDC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_classe_docto_mp"
        },
        {
            "table_oracle": "MCPR.MCPR_CLASSE_MATERIA_DOCTO",
            "pk_table_oracle": "CLMD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_classe_materia_docto"
        },
        {
            "table_oracle": "MCPR.MCPR_CNMP_CLASSE",
            "pk_table_oracle": "RCSS_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_cnmp_classe"
        },
        {
            "table_oracle": "MCPR.MCPR_CNMP_CLASSE_EXC",
            "pk_table_oracle": "RCEX_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_cnmp_classe_exc"
        },
        {
            "table_oracle": "MCPR.MCPR_CNMP_COLUNA",
            "pk_table_oracle": "CMCO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_cnmp_coluna"
        },
        {
            "table_oracle": "MCPR.MCPR_CNMP_COLUNA_TP_ANDAM",
            "pk_table_oracle": "RCND_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_cnmp_coluna_tp_andam"
        },
        {
            "table_oracle": "MCPR.MCPR_CNMP_LINHA",
            "pk_table_oracle": "CNLN_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_cnmp_linha"
        },
        {
            "table_oracle": "MCPR.MCPR_CNMP_LINHA_ASSUNTO",
            "pk_table_oracle": "LNAS_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_cnmp_linha_assunto"
        },
        {
            "table_oracle": "MCPR.MCPR_CNMP_SALDO_INICIAL",
            "pk_table_oracle": "RESD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_cnmp_saldo_inicial"
        },
        {
            "table_oracle": "MCPR.MCPR_CNMP_VALORES",
            "pk_table_oracle": "CNVL_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_cnmp_valores"
        },
        {
            "table_oracle": "MCPR.MCPR_CONTATO_PESSOA",
            "pk_table_oracle": "CONP_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_contato_pessoa"
        },
        {
            "table_oracle": "MCPR.MCPR_CONTROLE_AUTENTICACAO",
            "pk_table_oracle": "COAU_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_controle_autenticacao"
        },
        {
            "table_oracle": "MCPR.MCPR_CONTROLE_DISTRIBUICAO",
            "pk_table_oracle": "CNDI_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_controle_distribuicao"
        },
        {
            "table_oracle": "MCPR.MCPR_COR_PELE",
            "pk_table_oracle": "CORP_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_cor_pele"
        },
        {
            "table_oracle": "MCPR.MCPR_CORRELACIONAMENTO",
            "pk_table_oracle": "CORR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_correlacionamento"
        },
        {
            "table_oracle": "MCPR.MCPR_CORRELACIONAMENTO_UK",
            "pk_table_oracle": "CORR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_correlacionamento_uk"
        },
        {
            "table_oracle": "MCPR.MCPR_CURADOR",
            "pk_table_oracle": "CRDR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_curador"
        },
        {
            "table_oracle": "MCPR.MCPR_CURATELA",
            "pk_table_oracle": "CRLA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_curatela"
        },
        {
            "table_oracle": "MCPR.MCPR_CURATELA_CURADOR",
            "pk_table_oracle": "CURD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_curatela_curador"
        },
        {
            "table_oracle": "MCPR.MCPR_CURATELADO",
            "pk_table_oracle": "CRTL_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_curatelado"
        },
        {
            "table_oracle": "MCPR.MCPR_DOCUMENTO_CARGA_PESSOAL",
            "pk_table_oracle": "MDCP_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_documento_carga_pessoal"
        },
        {
            "table_oracle": "MCPR.MCPR_DOCUMENTO_VDM",
            "pk_table_oracle": "DVDM_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_documento_vdm"
        },
        {
            "table_oracle": "MCPR.MCPR_DOCUMENTO_VDM_FATOR",
            "pk_table_oracle": "DVDF_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_documento_vdm_fator"
        },
        {
            "table_oracle": "MCPR.MCPR_DOCUM_INFOR_ADICIONAL",
            "pk_table_oracle": "DIAD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_docum_infor_adicional"
        },
        {
            "table_oracle": "MCPR.MCPR_EMPRESTIMO",
            "pk_table_oracle": "MPEM_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_emprestimo"
        },
        {
            "table_oracle": "MCPR.MCPR_ENDERECO_DOCUMENTO",
            "pk_table_oracle": "EDOC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_endereco_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_ENDERECO_PESSOA",
            "pk_table_oracle": "ENPE_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_endereco_pessoa"
        },
        {
            "table_oracle": "MCPR.MCPR_ENDERECOS",
            "pk_table_oracle": "ENDC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_enderecos"
        },
        {
            "table_oracle": "MCPR.MCPR_ENQUADRA",
            "pk_table_oracle": "ENQU_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_enquadra"
        },
        {
            "table_oracle": "MCPR.MCPR_ESCOLARIDADE",
            "pk_table_oracle": "ESCO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_escolaridade"
        },
        {
            "table_oracle": "MCPR.MCPR_ESTADO_CIVIL",
            "pk_table_oracle": "ECIV_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_estado_civil"
        },
        {
            "table_oracle": "MCPR.MCPR_FAIXA_RENDA",
            "pk_table_oracle": "REND_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_faixa_renda"
        },
        {
            "table_oracle": "MCPR.MCPR_FASES_DOCUMENTO",
            "pk_table_oracle": "FSDC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_fases_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_FATOR_EXACERBADOR",
            "pk_table_oracle": "FAEX_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_fator_exacerbador"
        },
        {
            "table_oracle": "MCPR.MCPR_FILTRO_FUNCIONALIDADE",
            "pk_table_oracle": "FIFU_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_filtro_funcionalidade"
        },
        {
            "table_oracle": "MCPR.MCPR_FINALIDADE_VISTA",
            "pk_table_oracle": "FINA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_finalidade_vista"
        },
        {
            "table_oracle": "MCPR.MCPR_GRUPO_ANUARIO",
            "pk_table_oracle": "GRRA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_grupo_anuario"
        },
        {
            "table_oracle": "MCPR.MCPR_GRUPO_ORGAO_ANUARIO",
            "pk_table_oracle": "GOEA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_grupo_orgao_anuario"
        },
        {
            "table_oracle": "MCPR.MCPR_GUIA_TJ",
            "pk_table_oracle": "GUTJ_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_guia_tj"
        },
        {
            "table_oracle": "MCPR.MCPR_HISTORICO_CLASSE_DOC",
            "pk_table_oracle": "HCDC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_historico_classe_doc"
        },
        {
            "table_oracle": "MCPR.MCPR_HISTORICO_MATE_DOCU",
            "pk_table_oracle": "HMDO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_historico_mate_docu"
        },
        {
            "table_oracle": "MCPR.MCPR_HISTORICO_PROCEDENCIA_DOC",
            "pk_table_oracle": "HPCD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_historico_procedencia_doc"
        },
        {
            "table_oracle": "MCPR.MCPR_HISTORICO_RESP_DOCUMENTO",
            "pk_table_oracle": "HRDC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_historico_resp_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_HISTORICO_TIPO_DOC",
            "pk_table_oracle": "HTPD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_historico_tipo_doc"
        },
        {
            "table_oracle": "MCPR.MCPR_INFORMACAO_ADICIONAL",
            "pk_table_oracle": "INFD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_informacao_adicional"
        },
        {
            "table_oracle": "MCPR.MCPR_ITEM_GUIA_TJ",
            "pk_table_oracle": "ITGU_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_item_guia_tj"
        },
        {
            "table_oracle": "MCPR.MCPR_ITEM_MOVIMENTACAO",
            "pk_table_oracle": "ITEM_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_item_movimentacao"
        },
        {
            "table_oracle": "MCPR.MCPR_ITENS_FILTRO",
            "pk_table_oracle": "ITFI_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_itens_filtro"
        },
        {
            "table_oracle": "MCPR.MCPR_LEIS",
            "pk_table_oracle": "LEIS_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_leis"
        },
        {
            "table_oracle": "MCPR.MCPR_LINHA_ASSUNTO_RELAT",
            "pk_table_oracle": "RLAS_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_linha_assunto_relat"
        },
        {
            "table_oracle": "MCPR.MCPR_LINHA_DINAMICA",
            "pk_table_oracle": "TDIN_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_linha_dinamica"
        },
        {
            "table_oracle": "MCPR.MCPR_MATERIA_ASSUNTO",
            "pk_table_oracle": "MTAS_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_materia_assunto"
        },
        {
            "table_oracle": "MCPR.MCPR_MATERIA_GRUPO_DORJ",
            "pk_table_oracle": "MTDO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_materia_grupo_dorj"
        },
        {
            "table_oracle": "MCPR.MCPR_MATERIA_TP_DOCUMENTO",
            "pk_table_oracle": "MTTD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_materia_tp_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_MIGRA_DOCUMENTO",
            "pk_table_oracle": "MIGD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_migra_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_MODALIDADE_TP_DOCUMENTO",
            "pk_table_oracle": "MDTD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_modalidade_tp_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_MOVIMENTACAO",
            "pk_table_oracle": "MOVI_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_movimentacao"
        },
        {
            "table_oracle": "MCPR.MCPR_NACIONALIDADE",
            "pk_table_oracle": "NACI_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_nacionalidade"
        },
        {
            "table_oracle": "MCPR.MCPR_NIVEL_SIGILO",
            "pk_table_oracle": "NISI_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_nivel_sigilo"
        },
        {
            "table_oracle": "MCPR.MCPR_NOME_FALSO",
            "pk_table_oracle": "NFAL_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_nome_falso"
        },
        {
            "table_oracle": "MCPR.MCPR_NR_ORIGEM_DOCUMENTO",
            "pk_table_oracle": "NROR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_nr_origem_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_OCORRENCIA",
            "pk_table_oracle": "OCOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_ocorrencia"
        },
        {
            "table_oracle": "MCPR.MCPR_ORGAO_ANUARIO",
            "pk_table_oracle": "OREA_ORGI_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_orgao_anuario"
        },
        {
            "table_oracle": "MCPR.MCPR_ORGAO_APTO_ANDAMENTO",
            "pk_table_oracle": "OAPA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_orgao_apto_andamento"
        },
        {
            "table_oracle": "MCPR.MCPR_ORGAO_APTO_ARQUIVAMENTO",
            "pk_table_oracle": "OARQ_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_orgao_apto_arquivamento"
        },
        {
            "table_oracle": "MCPR.MCPR_ORGAO_PROTOCOLO",
            "pk_table_oracle": "OPRT_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_orgao_protocolo"
        },
        {
            "table_oracle": "MCPR.MCPR_PAPEL_OBRIGATORIO",
            "pk_table_oracle": "PAOB_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_papel_obrigatorio"
        },
        {
            "table_oracle": "MCPR.MCPR_PARAMETRO",
            "pk_table_oracle": "PMTS_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_parametro"
        },
        {
            "table_oracle": "MCPR.MCPR_PARTICIPACAO_EVENTO",
            "pk_table_oracle": "PTEV_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_participacao_evento"
        },
        {
            "table_oracle": "MCPR.MCPR_PATOLOGIAS_PESSOA_FISICA",
            "pk_table_oracle": "PAPF_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_patologias_pessoa_fisica"
        },
        {
            "table_oracle": "MCPR.MCPR_PERFIL",
            "pk_table_oracle": "PERF_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_perfil"
        },
        {
            "table_oracle": "MCPR.MCPR_PERSONAGEM",
            "pk_table_oracle": "PERS_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_personagem"
        },
        {
            "table_oracle": "MCPR.MCPR_PERSONAGEM_ANDAMENTO",
            "pk_table_oracle": "PEAN_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_personagem_andamento"
        },
        {
            "table_oracle": "MCPR.MCPR_PESSOA",
            "pk_table_oracle": "PESS_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_pessoa"
        },
        {
            "table_oracle": "MCPR.MCPR_PESSOA_FISICA",
            "pk_table_oracle": "PESF_PESS_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_pessoa_fisica"
        },
        {
            "table_oracle": "MCPR.MCPR_PESSOA_REQUERENTE",
            "pk_table_oracle": "PERE_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_pessoa_requerente"
        },
        {
            "table_oracle": "MCPR.MCPR_PRESTACAO_CONTA",
            "pk_table_oracle": "PRCO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_prestacao_conta"
        },
        {
            "table_oracle": "MCPR.MCPR_PROCEDENCIA_DOCUMENTO",
            "pk_table_oracle": "PCED_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_procedencia_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_PROFISSAO",
            "pk_table_oracle": "PROF_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_profissao"
        },
        {
            "table_oracle": "MCPR.MCPR_PROTOCOLO",
            "pk_table_oracle": "PRTL_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_protocolo"
        },
        {
            "table_oracle": "MCPR.MCPR_PROTOCOLO_TP_DOCUMENTO",
            "pk_table_oracle": "PTPD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_protocolo_tp_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_REGRA_ACOMP_DOCUMENTO",
            "pk_table_oracle": "REAC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_regra_acomp_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_REGRA_DISTRIBUICAO",
            "pk_table_oracle": "RGDI_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_regra_distribuicao"
        },
        {
            "table_oracle": "MCPR.MCPR_RELACIONAMENTO",
            "pk_table_oracle": "RELA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relacionamento"
        },
        {
            "table_oracle": "MCPR.MCPR_RELAT_ANUAL_TP_LINHA",
            "pk_table_oracle": "RATL_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relat_anual_tp_linha"
        },
        {
            "table_oracle": "MCPR.MCPR_RELAT_CORREG_ANALITICO",
            "pk_table_oracle": "RCAT_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relat_correg_analitico"
        },
        {
            "table_oracle": "MCPR.MCPR_RELAT_CORREG_MAT_2GRAU",
            "pk_table_oracle": "MRM2_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relat_correg_mat_2grau"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_ANUARIO",
            "pk_table_oracle": "RANU_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_anuario"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_ANUARIO_LINHA",
            "pk_table_oracle": "ALNH_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_anuario_linha"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_CNMP",
            "pk_table_oracle": "RCNP_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_cnmp"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_COLUNA",
            "pk_table_oracle": "RCLN_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_coluna"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_CORREGEDORIA",
            "pk_table_oracle": "RCOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_corregedoria"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_CORREG_LINHA",
            "pk_table_oracle": "MRCL_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_correg_linha"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_CORREG_MATERIA",
            "pk_table_oracle": "MRCM_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_correg_materia"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_CORREG_MENSAL",
            "pk_table_oracle": "RMEN_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_correg_mensal"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_LINHA",
            "pk_table_oracle": "RLNH_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_linha"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_ORGAO",
            "pk_table_oracle": "REOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_orgao"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_ORGAO_2GRAU",
            "pk_table_oracle": "RO2G_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_orgao_2grau"
        },
        {
            "table_oracle": "MCPR.MCPR_RELATORIO_VALORES",
            "pk_table_oracle": "RLVL_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_relatorio_valores"
        },
        {
            "table_oracle": "MCPR.MCPR_RENDA_BENEFICIO",
            "pk_table_oracle": "REBE_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_renda_beneficio"
        },
        {
            "table_oracle": "MCPR.MCPR_SIGILO_MATERIA",
            "pk_table_oracle": "SIMA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_sigilo_materia"
        },
        {
            "table_oracle": "MCPR.MCPR_SUB_ANDAMENTO",
            "pk_table_oracle": "STAO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_sub_andamento"
        },
        {
            "table_oracle": "MCPR.MCPR_SUBSCREVEM_COMUNICACAO",
            "pk_table_oracle": "SUCO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_subscrevem_comunicacao"
        },
        {
            "table_oracle": "MCPR.MCPR_SUB_TP_ANDAMENTO",
            "pk_table_oracle": "STPR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_sub_tp_andamento"
        },
        {
            "table_oracle": "MCPR.MCPR_TEMPLATE_TP_ANDAMENTO",
            "pk_table_oracle": "TEAN_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_template_tp_andamento"
        },
        {
            "table_oracle": "MCPR.MCPR_TEMPLATE_TP_DOCUMENTO",
            "pk_table_oracle": "TMDC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_template_tp_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_ANDAMENTO",
            "pk_table_oracle": "TPPR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_andamento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_ANDAMENTO_DOCUMENTO",
            "pk_table_oracle": "TPAD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_andamento_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_ANDAMENTO_ORGAO",
            "pk_table_oracle": "ANOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_andamento_orgao"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_ANDAMENTO_ORGAO_DEST",
            "pk_table_oracle": "TAOD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_andamento_orgao_dest"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_ANDTO_DOCTO_MATERIA",
            "pk_table_oracle": "TADM_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_andto_docto_materia"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_ANDTO_ENUNCIADO",
            "pk_table_oracle": "MTAE_ENCD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_andto_enunciado"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_ANDTO_ENUNCIADO",
            "pk_table_oracle": "MTAE_TPPR_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_andto_enunciado"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_AUTORIDADE",
            "pk_table_oracle": "TPAT_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_autoridade"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_CONCLUSOES_ATENDIMENTO",
            "pk_table_oracle": "TCNA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_conclusoes_atendimento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_CONTATO",
            "pk_table_oracle": "TPNT_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_contato"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_CORRELACIONAMENTO",
            "pk_table_oracle": "TPCO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_correlacionamento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_DOCUMENTO",
            "pk_table_oracle": "TPDC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_DOCUMENTO_ORGAO_EXT",
            "pk_table_oracle": "TDOE_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_documento_orgao_ext"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_ENDERECO",
            "pk_table_oracle": "TPEC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_endereco"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_ENVIO",
            "pk_table_oracle": "TPEN_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_envio"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_EVENTO",
            "pk_table_oracle": "TPEV_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_evento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_INVESTIMENTO",
            "pk_table_oracle": "INVS_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_investimento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_LOGRADOURO",
            "pk_table_oracle": "TPLO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_logradouro"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_MORADIA",
            "pk_table_oracle": "TPMO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_moradia"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_OCORRENCIA",
            "pk_table_oracle": "TPOC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_ocorrencia"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_PECA",
            "pk_table_oracle": "TPPA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_peca"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_PERSONAGEM",
            "pk_table_oracle": "TPPE_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_personagem"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_PESSOA_JURIDICA",
            "pk_table_oracle": "TPOE_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_pessoa_juridica"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_PROCESSO_TJ",
            "pk_table_oracle": "TPPC_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_processo_tj"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_REGRA_DISTRIBUICAO",
            "pk_table_oracle": "TPRD_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_regra_distribuicao"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_RELACIONAMENTO",
            "pk_table_oracle": "TPRE_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_relacionamento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_REMUNERACAO",
            "pk_table_oracle": "TPRM_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_remuneracao"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_RENDA",
            "pk_table_oracle": "TPRN_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_renda"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_SANGUINEO",
            "pk_table_oracle": "TPSG_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_sanguineo"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_SIT_ATEND_COMUNIC",
            "pk_table_oracle": "TPAM_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_sit_atend_comunic"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_SITUACAO_ANDAMENTO",
            "pk_table_oracle": "TPSA_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_situacao_andamento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_SITUACAO_DOCUMENTO",
            "pk_table_oracle": "TPST_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_situacao_documento"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_SITUACAO_GUIA",
            "pk_table_oracle": "TPGU_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_situacao_guia"
        },
        {
            "table_oracle": "MCPR.MCPR_TP_TEMPLATE",
            "pk_table_oracle": "TPTM_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_tp_template"
        },
        {
            "table_oracle": "MCPR.MCPR_TRANSF_CONTROL_INTERDICAO",
            "pk_table_oracle": "MTCI_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_transf_control_interdicao"
        },
        {
            "table_oracle": "MCPR.MCPR_VALOR_PARAMETRO",
            "pk_table_oracle": "VLPM_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_valor_parametro"
        },
        {
            "table_oracle": "MCPR.MCPR_VINCULO_AGRESSOR_VITIMA",
            "pk_table_oracle": "VIAG_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_vinculo_agressor_vitima"
        },
        {
            "table_oracle": "MCPR.MCPR_ACOMPANHAMENTO_DOCTO_LOG",
            "pk_table_oracle": "ACDL_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_acompanhamento_docto_log",
            "fields": "ACDL_ACRE_DK,ACDL_DK,ACDL_DS_ERRO,ACDL_DT_PROCESSAMENTO"
        },
        {
            "table_oracle": "MCPR.MCPR_ANDAMENTO",
            "pk_table_oracle": "PCAO_DK",
            "update_date_table_oracle": "",
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
            "table_oracle": "MCPR.MCPR_ATENDIMENTO",
            "pk_table_oracle": "ATND_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_atendimento",
            "fields":
            """
            ATND_ATND_DK_ORIGINARIO,ATND_DK,
            ATND_DT_ATENDIMENTO,ATND_DT_CANCELAMENTO,
            ATND_DT_CONCLUSAO,ATND_IN_TP_CONCLUSAO,
            ATND_MATE_DK_ATRIBUICAO,ATND_MOTIVO_CANCELAMENTO,
            ATND_NR_ATENDIMENTO,ATND_OBSERVACAO,
            ATND_ORGA_DK_ENCAMINHAMENTO,ATND_ORGI_ORGA_DK_RESPONSAVEL,
            ATND_PESF_DK_RESPONSAVEL,ATND_TCNA_DK
            """
        },
        {
            "table_oracle": "MCPR.MCPR_DADOS_COMUNICACAO",
            "pk_table_oracle": "DACO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_dados_comunicacao",
            "fields":
            """
            DACO_ATND_DK_REFERE,DACO_DK,
            DACO_DOCU_DK,DACO_DOCU_DK_REFERE,
            DACO_DS_REFERENCIA,DACO_DT_AR,
            DACO_DT_ENCERRAMENTO,DACO_DT_ESPERADA_RESPOSTA,
            DACO_DT_EXPEDICAO,DACO_DT_RECEBIMENTO,
            DACO_IN_INICIO_PRAZO_DT_AR,DACO_IN_OFICIO_CIRCULAR,
            DACO_IN_ORG_AGUARDA_RETORNO,DACO_IN_PGJ_ENVIA_AUTD_EXT,
            DACO_NR_DIAS_PRAZO,DACO_ORGA_DK_DESTINO,
            DACO_PCAO_DK_REFERE,DACO_PESS_DK_DESTINATARIO,
            DACO_TMPL_DK,DACO_TPAM_DK,
            DACO_TPEN_DK,DACO_TP_REFERENCIA
            """
        },
        {
            "table_oracle": "MCPR.MCPR_DOCUMENTO",
            "pk_table_oracle": "DOCU_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_documento",
            "fields":
            """
            DOCU_ANO,DOCU_ANO_DISTRIBUICAO,
            DOCU_ATND_DK,DOCU_CLDC_DK,DOCU_COMN_DK,
            DOCU_COMPLEMENTO_PROTOCOLO,DOCU_DESC_FATO_OLD3,
            DOCU_DK,DOCU_DT_CADASTRO,DOCU_DT_CANCELAMENTO,
            DOCU_DT_CARGA,DOCU_DT_DISTRIBUICAO,
            DOCU_DT_ENTRADA_MP,DOCU_DT_FATO,DOCU_DT_INICIAL,
            DOCU_DT_INICIO_PRAZO_ANDAMENTO,DOCU_DT_MIGRACAO,
            DOCU_DT_ULTIMA_TRANSMISSAO,DOCU_FSDC_DK,
            DOCU_IN_ANONIMO,DOCU_IN_APREENDIDO,
            DOCU_IN_ATENDIMENTO_OFICIO,DOCU_IN_DOCUMENTO_ELETRONICO,
            DOCU_IN_ENCAMINHAR_PGJ,DOCU_IN_HORARIO_FATO,
            DOCU_IN_IDOSO,DOCU_IN_PRESO,DOCU_IN_PRIORIDADE_INFANCIA,
            DOCU_IN_RELATADO,DOCU_IN_URGENTE,DOCU_ITEM_DK,
            DOCU_LOCALIZACAO_INTERNA,DOCU_MATE_DK,
            DOCU_MDTD_DK,DOCU_MOTIVO_CANCELAMENTO,
            DOCU_NISI_DK,DOCU_NM_PARTE_PROTOCOLO,DOCU_NR_CIAC,
            DOCU_NR_DIAS_PRAZO_ANDAMENTO,DOCU_NR_DIAS_PRAZO_DOCUMENTO,
            DOCU_NR_DISTRIBUICAO,DOCU_NR_DOCUMENTO_MIGRACAO,
            DOCU_NR_EXTERNO,DOCU_NR_FOLHAS,DOCU_NR_MP,
            DOCU_NR_TJ,DOCU_OB_SGP,DOCU_ORGA_DK_ORIGEM,
            DOCU_ORGE_ORGA_DK_DELEG_FATO,DOCU_ORGE_ORGA_DK_DELEG_ORIGEM,
            DOCU_ORGE_ORGA_DK_VARA,DOCU_ORGI_ORGA_DK_CARGA,
            DOCU_ORGI_ORGA_DK_ENTRADA,DOCU_ORGI_ORGA_DK_RESPONSAVEL,
            DOCU_PCAO_DK,DOCU_PCED_DK_PROCEDENCIA,
            DOCU_PESF_PESS_DK_RESP_INSTAUR,DOCU_PORTARIA,
            DOCU_PRTL_DK,DOCU_REQUERENTE_PROTOCOLO,
            DOCU_TPDC_DK,DOCU_TPPC_DK,DOCU_TPPR_DK,
            DOCU_TPST_DK,DOCU_TX_ETIQUETA,DOCU_VIST_DK_ABERTA,
            DOCU_VL_CAUSA,DOCU_VL_DANO,DOCU_VOLUMES
            """
        },
        {
            "table_oracle": "MCPR.MCPR_FOTO",
            "pk_table_oracle": "FOTO_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_foto",
            "fields":
            """
            FOTO_DATA_CADASTRO,FOTO_DATA_FOTO,
            FOTO_DK,FOTO_DS_CONTEUDO,
            FOTO_NM_ARQUIVO,FOTO_PESF_PESS_DK,
            FOTO_TAMANHO_OBJETO,FOTO_TP_ARQUIVO_OBJETO
            """
        },
        {
            "table_oracle": "MCPR.MCPR_HISTORICO_FASE_DOC",
            "pk_table_oracle": "HCFS_DK",
            "update_date_table_oracle": "",
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
            "table_oracle": "MCPR.MCPR_OBJETOS_ANEXOS",
            "pk_table_oracle": "OBJX_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_objetos_anexos",
            "fields":
            """
            OBJX_ATND_DK,OBJX_DK,
            OBJX_DOCU_DK,OBJX_DS_CONTEUDO,
            OBJX_DT_ANEXO,OBJX_DT_OBJETO,
            OBJX_NM_ARQUIVO,OBJX_PCAO_DK,
            OBJX_PESS_DK,OBJX_TAMANHO_OBJETO,
            OBJX_TP_ARQUIVO_OBJETO,OBJX_TPPA_DK
            """
        },
        {
            "table_oracle": "MCPR.MCPR_PESSOA_JURIDICA",
            "pk_table_oracle": "PESJ_PESS_DK",
            "update_date_table_oracle": "",
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
            "table_oracle": "MCPR.MCPR_SOL_REL_EST_ELEITORAL",
            "pk_table_oracle": "SREE_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_sol_rel_est_eleitoral",
            "fields":
            """
            SREE_ANO_MES_RELAT,SREE_DK,
            SREE_DT_GERACAO,SREE_DT_SOLICITACAO,
            SREE_LOGIN,SREE_NM_ARQUIVO
            """
        },
        {
            "table_oracle": "MCPR.MCPR_TEMPLATES",
            "pk_table_oracle": "TMPL_DK",
            "update_date_table_oracle": "",
            "table_hive": "mcpr_templates",
            "fields":
            "TMPL_DK,TMPL_IN_ATIVO,TMPL_NM_TEMPLATE,TMPL_ORGI_DK,TMPL_TPTM_DK"
        },
        {
            "table_oracle": "MCPR.MCPR_VISTA",
            "pk_table_oracle": "VIST_DK",
            "update_date_table_oracle": "",
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
        # Schema RH
        {
            "table_oracle": "RH.FUNCIONARIO",
            "pk_table_oracle": "CDMATRICULA",
            "update_date_table_oracle": "",
            "table_hive": "rh_funcionario"
        },
        {
            "table_oracle": "RH.HIST_FUNC",
            "pk_table_oracle": "HFUNC_SEQ",
            "update_date_table_oracle": "",
            "table_hive": "rh_hist_func"
        },
        # Schema ORGI
        {
            "table_oracle": "ORGI.ORGI_AREA_ATUACAO",
            "pk_table_oracle": "ARAT_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_area_atuacao"
        },
        {
            "table_oracle": "ORGI.ORGI_AREA_ATUACAO_CNMP",
            "pk_table_oracle": "OAAC_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_area_atuacao_cnmp"
        },
        {
            "table_oracle": "ORGI.ORGI_AUXILIA",
            "pk_table_oracle": "ORAU_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_auxilia"
        },
        {
            "table_oracle": "ORGI.ORGI_CAOP",
            "pk_table_oracle": "CAOP_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_caop"
        },
        {
            "table_oracle": "ORGI.ORGI_CAOP_MATERIA",
            "pk_table_oracle": "CAMT_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_caop_materia"
        },
        {
            "table_oracle": "ORGI.ORGI_CAOP_ORGAO",
            "pk_table_oracle": "CAOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_caop_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_CARACTERISTICA",
            "pk_table_oracle": "OTPC_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_caracteristica"
        },
        {
            "table_oracle": "ORGI.ORGI_CARACTERISTICA_ORGAO",
            "pk_table_oracle": "OCAR_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_caracteristica_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_COMARCA",
            "pk_table_oracle": "CMRC_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_comarca"
        },
        {
            "table_oracle": "ORGI.ORGI_COMARCA_CIDADE",
            "pk_table_oracle": "COCI_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_comarca_cidade"
        },
        {
            "table_oracle": "ORGI.ORGI_CORRESP_EXT",
            "pk_table_oracle": "COEX_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_corresp_ext"
        },
        {
            "table_oracle": "ORGI.ORGI_EMAIL_ORGAO",
            "pk_table_oracle": "EMOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_email_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_ENDERECO",
            "pk_table_oracle": "EDCO_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_endereco"
        },
        {
            "table_oracle": "ORGI.ORGI_ENDERECO_COMPLEMENTO",
            "pk_table_oracle": "ECMP_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_endereco_complemento"
        },
        {
            "table_oracle": "ORGI.ORGI_ENDERECO_ORGAO",
            "pk_table_oracle": "ENOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_endereco_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_ENDERECO_TELEFONE",
            "pk_table_oracle": "ENDT_ENOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_endereco_telefone"
        },
        {
            "table_oracle": "ORGI.ORGI_ENDERECO_TELEFONE",
            "pk_table_oracle": "ENDT_TLFO_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_endereco_telefone"
        },
        {
            "table_oracle": "ORGI.ORGI_FATOR_DE_CUSTO",
            "pk_table_oracle": "FCST_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_fator_de_custo"
        },
        {
            "table_oracle": "ORGI.ORGI_FORO",
            "pk_table_oracle": "COFO_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_foro"
        },
        {
            "table_oracle": "ORGI.ORGI_FORO_ORGAO",
            "pk_table_oracle": "FORG_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_foro_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_GRUPO_PREF",
            "pk_table_oracle": "GRPF_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_grupo_pref"
        },
        {
            "table_oracle": "ORGI.ORGI_HISTORICO_NM_ORGAO",
            "pk_table_oracle": "HNMO_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_historico_nm_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_LOCAL_ATUACAO",
            "pk_table_oracle": "LATU_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_local_atuacao"
        },
        {
            "table_oracle": "ORGI.ORGI_LOCAL_ATUACAO_ASSUNTO",
            "pk_table_oracle": "ASAT_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_local_atuacao_assunto"
        },
        {
            "table_oracle": "ORGI.ORGI_MATERIA_ORGAO",
            "pk_table_oracle": "MAOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_materia_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_NUCLEO",
            "pk_table_oracle": "NUCL_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_nucleo"
        },
        {
            "table_oracle": "ORGI.ORGI_NUCLEO_ORGAO",
            "pk_table_oracle": "NUOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_nucleo_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_ORGAO",
            "pk_table_oracle": "ORGI_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_REGIAO",
            "pk_table_oracle": "REGI_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_regiao"
        },
        {
            "table_oracle": "ORGI.ORGI_REGIAO_COMARCA",
            "pk_table_oracle": "RECM_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_regiao_comarca"
        },
        {
            "table_oracle": "ORGI.ORGI_RELACIONAMENTO_ORGAO",
            "pk_table_oracle": "RELO_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_relacionamento_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_SITUACAO",
            "pk_table_oracle": "SITU_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_situacao"
        },
        {
            "table_oracle": "ORGI.ORGI_SUBTIPO",
            "pk_table_oracle": "ORST_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_subtipo"
        },
        {
            "table_oracle": "ORGI.ORGI_TELEFONE",
            "pk_table_oracle": "TLFO_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_telefone"
        },
        {
            "table_oracle": "ORGI.ORGI_TP_COMPLEMENTO",
            "pk_table_oracle": "TPCP_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_tp_complemento"
        },
        {
            "table_oracle": "ORGI.ORGI_TP_IMOVEL",
            "pk_table_oracle": "TPIM_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_tp_imovel"
        },
        {
            "table_oracle": "ORGI.ORGI_TP_LOCAL_ATUACAO",
            "pk_table_oracle": "TPLA_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_tp_local_atuacao"
        },
        {
            "table_oracle": "ORGI.ORGI_TP_ORGAO",
            "pk_table_oracle": "TPOR_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_tp_orgao"
        },
        {
            "table_oracle": "ORGI.ORGI_TP_PROPRIEDADE_IMOVEL",
            "pk_table_oracle": "TMVL_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_tp_propriedade_imovel"
        },
        {
            "table_oracle": "ORGI.ORGI_TP_RELACIONAMENTO",
            "pk_table_oracle": "TIRE_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_tp_relacionamento"
        },
        {
            "table_oracle": "ORGI.ORGI_ZONA_ELEITORAL",
            "pk_table_oracle": "ZELE_DK",
            "update_date_table_oracle": "",
            "table_hive": "orgi_zona_eleitoral"
        },
        {
            "table_oracle": "ORGI.ORGI_VW_ORGAO_LOCAL_ATUAL",
            "pk_table_oracle": "orlw_dk",
            "update_date_table_oracle": "",
            "table_hive": "orgi_vw_orgao_local_atual"
        },
        {
            "table_oracle": "MMPS.MMPS_VW_RH_FOTO_FUNC",
            "pk_table_oracle": "VTFU_CDMATRICULA",
            "update_date_table_oracle": "",
            "table_hive": "mmps_vw_rh_foto_func"
        },
        {
            "table_oracle": "MMPS.MMPS_ADM_RH_MOV_PROM",
            "pk_table_oracle": "MMPM_MATRICULA",
            "update_date_table_oracle": "",
            "table_hive": "mmps_adm_rh_mov_prom"
        },
        {
            "table_oracle": "MMPS.MMPS_ADM_RH_MOV_ELEIT",
            "pk_table_oracle": "MMPE_ORDEM",
            "update_date_table_oracle": "",
            "table_hive": "mmps_adm_rh_mov_eleit"
        },
        {
            "table_oracle": "MMPS.MMPS_ADM_RH_MOV_SERV",
            "pk_table_oracle": "MMSV_MATRICULA",
            "update_date_table_oracle": "",
            "table_hive": "mmps_adm_rh_mov_serv"
        },
        {
            "table_oracle": "MMPS.MMPS_ADM_RH_MOV_PROC",
            "pk_table_oracle": "MMPC_MATRICULA",
            "update_date_table_oracle": "",
            "table_hive": "mmps_adm_rh_mov_proc"
        }
    ]
}
"""
->  Query to generate the dict parameters
    for tables that don't have columns blob or clob.
    Theses tables will be load with all
    columns and doesn't need parameter fields
    SELECT 	'{
        "table_oracle": ' AS label_table_oracle,
        '"' || cons.owner || '.' || cols.table_name || '",'
        as table_oracle,
        '"pk_table_oracle": ' AS label_pk_table_oracle,
        '"' || cols.column_name || '",' AS pk_table_oracle,
        "update_date_table_oracle": '
        AS label_update_date_table_oracle,
        '"' || '' || '",' AS table_hive,
        '"table_hive": ' AS label_table_hive,
        '"' || LOWER(cols.table_name) || '" },' AS table_hive
    FROM all_constraints cons
    JOIN all_cons_columns cols
    ON cons.constraint_name = cols.constraint_name
    AND cons.owner = cols.owner
    JOIN (
        SELECT  table_name
        FROM all_tab_columns
        WHERE owner = 'MCPR' AND table_name NOT IN (
            SELECT TABLE_NAME FROM all_tab_columns
            WHERE owner = 'MCPR' AND (data_type = 'BLOB' OR data_type = 'CLOB')
        )
        GROUP BY table_name
    ) p
    ON cols.table_name = p.TABLE_NAME
    WHERE cons.constraint_type = 'P' AND cons.owner = 'MCPR'
    ORDER BY cols.table_name, cols.POSITION


->  Query to generate the dict parameters
    for tables that start with 'A' and dont have
    explicit primary key like constraint
    in table configuration (Just for schema MCPR)

    SELECT 	'{
    "table_oracle": ' AS label_table_oracle,
    '"' || cons.owner || '.' || cols.table_name || '",' AS table_oracle,
    '"pk_table_oracle": ' AS label_pk_table_oracle,
    '"",' AS pk_table_oracle,
    "update_date_table_oracle": ' AS label_update_date_table_oracle,
    '"' || '' || '",' AS update_date_table_oracle,
    '"table_hive": ' AS label_table_hive,
    '"' || LOWER(REPLACE(cols.table_name, '$', '_')) || '" },' AS table_hive
    FROM all_constraints cons
    JOIN all_cons_columns cols
    ON cons.constraint_name = cols.constraint_name AND cons.owner = cols.owner
    WHERE  cons.owner = 'MCPR' AND cols.table_name LIKE 'A%'
    GROUP BY cols.table_name, cons.owner
    ORDER BY cols.table_name


->  Query to generate the params for tables
    that have columns blob or clob.
    Theses tables will be load without
    columns blob or clob and need parameter fields

    SELECT 	'{
    "table_oracle": ' AS label_table_oracle,
    '"' || cons.owner || '.' || cols.table_name || '",' AS table_oracle,
    '"pk_table_oracle": ' AS label_pk_table_oracle,
    '"' || cols.column_name || '",' AS pk_table_oracle,
    "update_date_table_oracle": ' AS label_update_date_table_oracle,
    '"' || '' || '",' AS update_date_table_oracle,
    '"table_hive": ' AS th,
    '"' || LOWER(cols.table_name) || '",'  AS table_hive,
    '"fields" : ' AS qr, '"' || p.elements || '" },' AS fields
    FROM all_constraints cons
    JOIN all_cons_columns cols
    ON cons.constraint_name = cols.constraint_name AND cons.owner = cols.owner
    JOIN (
        SELECT TABLE_NAME,
            LTRIM(MAX(SYS_CONNECT_BY_PATH(COLUMN_NAME,','))
            KEEP (DENSE_RANK LAST ORDER BY curr),',') AS elements
        FROM   (
            SELECT TABLE_NAME,
                COLUMN_NAME,
                ROW_NUMBER() OVER
                (PARTITION BY TABLE_NAME ORDER BY COLUMN_NAME) AS curr,
                ROW_NUMBER() OVER
                (PARTITION BY TABLE_NAME ORDER BY COLUMN_NAME) -1 AS prev
            FROM  all_tab_columns
            WHERE table_name in
            (SELECT TABLE_NAME
            FROM all_tab_columns
            WHERE owner = 'MCPR'
            AND (data_type = 'BLOB' OR data_type = 'CLOB'))
            AND owner = 'MCPR'AND data_type NOT IN ('BLOB','CLOB')
        )
        GROUP BY TABLE_NAME
        CONNECT BY prev = PRIOR curr AND TABLE_NAME = PRIOR TABLE_NAME
        START WITH curr = 1
    ) p
    ON cols.table_name = p.TABLE_NAME
    WHERE cons.constraint_type = 'P' AND cons.owner = 'MCPR'
    ORDER BY cols.table_name, cols.POSITION
"""
