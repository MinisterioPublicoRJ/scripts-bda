#-*-coding:utf-8-*-
import argparse
import difflib
import pyspark
import re
import unicodedata

from datetime import datetime, timedelta
from generic_utils import execute_compute_stats

def name_similarity(name_left, name_right):
    if not name_left or not name_right:
        return 0

    def remove_accents_n(value):
        text = unicodedata.normalize('NFD', value)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return text

    name_left = remove_accents_n(re.sub(r"\s+", "", name_left or ""))
    name_right = remove_accents_n(re.sub(r"\s+", "", name_right or ""))
    return difflib.SequenceMatcher(None, name_left, name_right).ratio()


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_andamento_investigacoes") \
            .enableHiveSupport() \
            .getOrCreate()

    spark.udf.register("name_similarity", name_similarity)

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']
    personagens_cutoff = options['personagens_cutoff']
    nb_past_days = options['nb_past_days']
    LIMIAR_SIMILARIDADE = options["limiar_similaridade"]
    table_name = options['table_name']

    dt_inicio = datetime.now() - timedelta(nb_past_days)

    REGEX_EXCLUSAO_ORGAOS = (
        "(^MP$|MINIST[EÉ]RIO P[UÚ]BLICO|DEFENSORIA P[UÚ]BLICA"
        "|MINSTERIO PUBLICO|MPRJ|MINITÉRIO PÚBLICO|JUSTI[ÇC]A P[UÚ]BLICA)"
    )

    spark.sql(
        """
        SELECT
            docu_orgi_orga_dk_responsavel as orgao_dk,
            cldc_ds_classe as classe_documento,
            DOCU_NR_MP,
            DOCU_NR_EXTERNO,
            docu_tx_etiqueta,
            PCAO_DT_ANDAMENTO,
            TPPR_DESCRICAO,
            TPPR_DK,
            DOCU_DK
        FROM {0}.MCPR_DOCUMENTO
        JOIN {0}.MCPR_VISTA ON VIST_DOCU_DK = DOCU_DK
        JOIN {0}.MCPR_ANDAMENTO ON PCAO_VIST_DK = VIST_DK
        JOIN {0}.MCPR_SUB_ANDAMENTO ON STAO_PCAO_DK = PCAO_DK
        JOIN {0}.MCPR_TP_ANDAMENTO ON TPPR_DK = STAO_TPPR_DK
        JOIN {1}.atualizacao_pj_pacote ON docu_orgi_orga_dk_responsavel = id_orgao
        JOIN {1}.tb_regra_negocio_investigacao
            ON cod_pct = cod_atribuicao
            AND classe_documento = docu_cldc_dk
        JOIN {0}.mcpr_classe_docto_mp ON cldc_dk = docu_cldc_dk
        WHERE pcao_dt_cancelamento IS NULL
        AND docu_tpst_dk != 11
        AND docu_fsdc_dk = 1
        """.format(schema_exadata, schema_exadata_aux)
    ).createOrReplaceTempView('DOCU_TOTAIS')

    spark.sql(
        """
            WITH PERSONAGENS AS (SELECT
                docu_nr_mp,
                pess_dk,
                CASE WHEN concat_ws(', ', collect_list(cast(tppe_dk as int))) REGEXP '(^| )(290|7|21|317|20|14|32|345|40|5|24)(,|$)' THEN 1 ELSE 0 END AS is_investigado,
                pess_nm_pessoa,
                LEAD(pess_nm_pessoa) OVER (PARTITION BY docu_nr_mp ORDER BY pess_nm_pessoa) proximo_nome
            FROM DOCU_TOTAIS
            JOIN {0}.mcpr_personagem ON pers_docu_dk = docu_dk
            JOIN {0}.mcpr_pessoa ON pers_pess_dk = pess_dk
            JOIN {0}.mcpr_tp_personagem ON pers_tppe_dk = tppe_dk
            AND (
                tppe_dk <> 7 OR
                pess_nm_pessoa not rlike '{REGEX_EXCLUSAO_ORGAOS}'
            )
            GROUP BY docu_nr_mp, pess_nm_pessoa, pess_dk -- remove duplicação de personagens com mesmo
        ),
        PERSONAGENS_SIMILARIDADE AS (
        SELECT docu_nr_mp,
        pess_dk,
        pess_nm_pessoa,
        is_investigado,
            CASE
                WHEN name_similarity(pess_nm_pessoa, proximo_nome) > {LIMIAR_SIMILARIDADE} THEN false
                ELSE true
            END AS primeira_aparicao
            FROM PERSONAGENS
        )
        SELECT
            docu_nr_mp,
            pess_nm_pessoa,
            B.representante_dk,
            row_number() OVER (PARTITION BY docu_nr_mp ORDER BY B.has_dk DESC, A.is_investigado DESC) as nr_pers
        FROM PERSONAGENS_SIMILARIDADE A
        LEFT JOIN (
            SELECT pess_dk, R.representante_dk, 1 as has_dk FROM {1}.test_tb_pip_investigados_representantes R
            JOIN
            (
                SELECT DISTINCT representante_dk
                FROM {1}.test_tb_pip_investigados_representantes
                JOIN {0}.mcpr_personagem ON pess_dk = pers_pess_dk
                WHERE pers_tppe_dk IN (290, 7, 21, 317, 20, 14, 32, 345, 40, 5, 24)
            ) RI ON RI.representante_dk = R.representante_dk
        ) B ON A.pess_dk = B.pess_dk
        WHERE primeira_aparicao = true
        """.format(
            schema_exadata, schema_exadata_aux,
            REGEX_EXCLUSAO_ORGAOS=REGEX_EXCLUSAO_ORGAOS,
            LIMIAR_SIMILARIDADE=LIMIAR_SIMILARIDADE
        )
    ).createOrReplaceTempView("PERSONAGENS_SIMILARIDADE")

    spark.sql(
        """
        SELECT
            docu_nr_mp,
            concat_ws(', ', collect_list(nm_personagem)) as personagens,
            MAX(representante_dk) as representante_dk
        FROM (
            SELECT
                docu_nr_mp,
                CASE
                    WHEN nr_pers = {1} THEN 'e outros...'
                    ELSE pess_nm_pessoa END
                AS nm_personagem,
                nr_pers,
                CASE
                    WHEN nr_pers = 1 THEN representante_dk
                    ELSE NULL END
                AS representante_dk
            FROM
            PERSONAGENS_SIMILARIDADE
            WHERE nr_pers <= {1})
        GROUP BY docu_nr_mp
        """.format(schema_exadata, personagens_cutoff)
    ).createOrReplaceTempView('DOCU_PERSONAGENS')

    spark.sql(
        """
        SELECT
            DOCU_NR_MP,
            MAX(PCAO_DT_ANDAMENTO) AS DT_ULTIMO
        FROM DOCU_TOTAIS
        GROUP BY DOCU_NR_MP
        """
    ).createOrReplaceTempView('DTS_ULTIMOS_ANDAMENTOS')


    # url_tjrj sempre NULL, para poder aproveitar o mesmo formato dos dados no front
    lista_investigacoes = spark.sql(
        """
        SELECT
            A.orgao_dk,
            A.classe_documento as cldc_dk,
            A.docu_nr_mp,
            A.docu_nr_externo,
            A.docu_tx_etiqueta,
            P.personagens,
            P.representante_dk,
            A.pcao_dt_andamento as dt_ultimo_andamento,
            concat_ws(', ', collect_list(A.tppr_descricao)) as ultimo_andamento,
            CAST(NULL as string) as url_tjrj
        FROM DOCU_TOTAIS A
        JOIN DTS_ULTIMOS_ANDAMENTOS ULT
            ON A.DOCU_NR_MP = ULT.DOCU_NR_MP
            AND A.PCAO_DT_ANDAMENTO = ULT.DT_ULTIMO
        JOIN DOCU_PERSONAGENS P ON P.DOCU_NR_MP = A.DOCU_NR_MP
        GROUP BY A.orgao_dk, A.classe_documento, A.docu_nr_mp,
            A.docu_nr_externo, A.docu_tx_etiqueta, P.personagens, P.representante_dk,
            A.pcao_dt_andamento
        """
    )

    table_name = "{}.test_{}".format(schema_exadata_aux, table_name)

    lista_investigacoes.write.mode("overwrite").saveAsTable("temp_table_lista_investigacoes")
    temp_table = spark.table("temp_table_lista_investigacoes")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_lista_investigacoes")

    execute_compute_stats(table_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table lista investigacoes")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    parser.add_argument('-c','--personagensCutoff', metavar='personagensCutoff', type=int, default=2, help='')
    parser.add_argument('-p','--nbPastDays', metavar='nbPastDays', type=int, default=7, help='')
    parser.add_argument('-l','--limiarSimilaridade', metavar='limiarSimilaridade', type=float, default=0.85, help='')
    parser.add_argument('-t','--tableName', metavar='tableName', type=str, help='')
    
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata,
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort,
                    'personagens_cutoff' : args.personagensCutoff,
                    'nb_past_days': args.nbPastDays,
                    'limiar_similaridade': args.limiarSimilaridade,
                    'table_name' : args.tableName,
                }

    execute_process(options)
