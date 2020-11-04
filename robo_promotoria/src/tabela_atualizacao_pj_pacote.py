#-*-coding:utf-8-*-
import argparse
import pyspark

from pyspark.sql.functions import unix_timestamp, from_unixtime, current_timestamp, lit, date_format
from generic_utils import execute_compute_stats

CUSTOM_MAPPINGS = [
    (200, "PIPs Territoriais 1a CI", (29934303, 29934337, 29926583, 29926616, 29926805, 29927047, 29933374, 29933418, 29933469, 29933470, 29933490, 29933502, 29933521, 29933590, 29933830, 29933850, 29933955, 29933967, 29933988, 29934004, 29934012, 29934277, 29934363, 29934376)),
    (201, "PIPs Territoriais 2a CI", (30061694, 30061723, 30061624, 30034384, 30061094)),
    (202, "PIPs Territoriais 3a CI", (30069669, 30069693, 30069732, 30070041, 30069167, 30069433, 30069453, 30069490, 30069516)),
    (203, "PIPs Territoriais Interior", (300977, 300987, 400731, 400758, 400736, 1249804, 5679985, 6115386, 7419262, 7419344)),
    (204, "PIPs Violência Doméstica 1a CI", (29941071, 29941061, 29934401, 29934732)),
    (205, "PIPs Violência Doméstica 2a CI", (30034671, 30061329)),
    (206, "PIPs Violência Doméstica 3a CI", (29941322, 30034664, 30069601)),
    (207, "PIPs Especializadas 1a CI", (29941099, 29941140, 29941222, 29941251)),
    (208, "PIPs Especializadas 2a CI", (29941368, 30061803)),
    (209, "PIPs Especializadas 3a CI", (30070783, 30070806)),
    (180, "Tutela Coletiva - Sistema Prisional e Estabelecimentos de Aplicação de Medidas Socioeducativas", (14038896, 19974413)),
    (181, "Tutela Coletiva - Proteção do Idoso e da Pessoa com Deficiência", (28933866, 19028031)),
    (182, "Tutela Coletiva - Prestacionais", (24222687, 27700444, 28006690)),
    (183, "Tutela Coletiva - Meio Ambiente, Consumidor e Saúde", (400652, 1342457)),
    (20, "Tutela Coletiva Ampla", (17216152,)),
    (24, "Tutela Coletiva - Meio Ambiente e Consumidor", (400564,)),
    (25, "Tutela Coletiva - Cidadania Ampla", (400540, 913962)),
]
LIST_TO_REMOVE = tuple(x for m in CUSTOM_MAPPINGS for x in m[2])

def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_atualizacao_pj_pacote") \
            .enableHiveSupport() \
            .getOrCreate()
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    main_query = """
        SELECT 
            cod_pct, pacote_de_atribuicao as pacote_atribuicao, cast(split(query_origem, ' ')[4] as int) as id_orgao,
            orgao_codamp,
            orgi_nm_orgao as orgi_nm_orgao
        FROM {OPENGEO_SCHEMA}.stat_pacote_atribuicao_2019
        LEFT JOIN {schema_exadata}.orgi_orgao ON orgi_dk = id_orgao
        WHERE cast(split(query_origem, ' ')[4] as int) NOT IN {LIST_TO_REMOVE}
    """.format(
        OPENGEO_SCHEMA='opengeo',
        LIST_TO_REMOVE=LIST_TO_REMOVE,
        schema_exadata=schema_exadata
    )

    union_queries = ""
    for mapping in CUSTOM_MAPPINGS:
        cod_pct = mapping[0]
        descricao = mapping[1]
        nb_rows = len(mapping[2])
        stack_args = tuple([nb_rows] + list(mapping[2]))

        union_queries += """
            UNION ALL
            SELECT 
                {cod_pct} as cod_pct,
                '{descricao}' as pacote_atribuicao,
                id_orgao,
                orgi_nm_orgao_abrev as orgao_codamp,
                orgi_nm_orgao as orgi_nm_orgao
            FROM (SELECT stack{stack_args} as id_orgao) t 
            JOIN {schema_exadata}.orgi_orgao ON orgi_dk = id_orgao
        """.format(
            cod_pct=cod_pct,
            descricao=descricao,
            stack_args=stack_args,
            schema_exadata=schema_exadata
        )

    query = main_query + union_queries

    table = spark.sql(query)

    table_name = options['table_name']
    table_name = "{}.{}".format(schema_exadata_aux, table_name)
    table.write.mode("overwrite").saveAsTable("temp_table_atualizacao_pj_pacote")
    temp_table = spark.table("temp_table_atualizacao_pj_pacote")
    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_atualizacao_pj_pacote")

    execute_compute_stats(table_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table acervo")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    parser.add_argument('-t','--tableName', metavar='tableName', default="atualizacao_pj_pacote", type=str, help='')
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort,
                    'table_name' : args.tableName,
                }

    execute_process(options)