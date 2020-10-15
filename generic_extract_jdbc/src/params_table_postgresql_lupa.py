params = {
    "driver": "org.postgresql.Driver",
    "schema_hdfs": "opengeo",
    "tables": [
        # Schema MCPR
        {
            "table_jdbc": "financeiro.financeiro",
            "pk_table_jdbc": "ID",
            "update_date_table_jdbc": "",
            "table_hive": "financeiro"
        },
        {
            "table_jdbc": 
            """
            (SELECT nome, cpf, localidade, cod_ibge::integer,
            ano_eleicao, foto, url_tse FROM lupa.governantes_rj) t """,
            "pk_table_jdbc": "cod_ibge",
            "update_date_table_jdbc": "",
            "table_hive": "lupa_governantes_rj"
        },
        {
            "table_jdbc": 
            """
            (SELECT cod_craai, nome_craai, municipio,
            cod_municipio::integer, gentilico,
            prefeito, vice_prefeito, site,
            telefone, endereco, fonte FROM lupa.prefeituras) t""",
            "pk_table_jdbc": "cod_municipio",
            "update_date_table_jdbc": "",
            "table_hive": "lupa_prefeituras"
        },
        {
            "table_jdbc": """(SELECT cod_municipio::integer,
            municipio, criacao, aniversario FROM lupa.dados_gerais_municipio) t""",
            "pk_table_jdbc": "cod_municipio",
            "update_date_table_jdbc": "",
            "table_hive": "lupa_dados_gerais_municipio"
        },
        {
            "table_jdbc": "lupa.orgaos_mprj",
            "pk_table_jdbc": "id",
            "update_date_table_jdbc": "",
            "table_hive": "lupa_orgaos_mprj"
        }
    ]
}