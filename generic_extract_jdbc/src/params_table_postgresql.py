params = {
    "driver": "org.postgresql.Driver",
    "tables": [
        # Schema MCPR
        {
            "table_jdbc": """
                (SELECT objectid::int, document_id::text, proc_id, proc_numero, depo_id, cod_cisp, depo_nome, depo_circ_id, 
                data_liberacao, depo_circ_nome, catg_id, catg_descricao, stts_id, stts_descricao, proc_data::timestamp
                FROM seg_pub.in_pol_procedimento) t""",
            "pk_table_jdbc": "objectid",
            "update_date_table_jdbc": "",
            "table_hive": "seg_pub_in_pol_procedimento"
        },
        {
            "table_jdbc": """
                (SELECT objectid::int, document_id::text, proc_id, proc_numero, ocor_id, end_logradouro, 
                end_numero, end_bairro, end_municipio, end_uf, endereco, tipo_delito, tipo_delito_desc, etit, agregado_isp, 
                ocor_dt_inicial::timestamp, ocor_hr_inicial, periodo, ocorcapitulacao
                FROM seg_pub.in_pol_ocorrencia) t""",
            "pk_table_jdbc": "objectid",
            "update_date_table_jdbc": "",
            "table_hive": "seg_pub_in_pol_ocorrencia"
        }
    ]
}