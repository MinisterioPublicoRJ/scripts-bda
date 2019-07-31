params = {
    "driver": "org.postgresql.Driver",
    "schema_hdfs": "opengeo",
    "tables": [
        # Schema MCPR
        {
            "table_oracle": "financeiro.financeiro",
            "pk_table_oracle": "ID",
            "update_date_table_oracle": "",
            "table_hive": "financeiro"
        }
    ]
}