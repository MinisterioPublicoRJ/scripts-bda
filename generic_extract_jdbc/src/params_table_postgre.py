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
            ano_eleicao, foto FROM lupa.governantes_rj) t """,
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
        }
    ]
}