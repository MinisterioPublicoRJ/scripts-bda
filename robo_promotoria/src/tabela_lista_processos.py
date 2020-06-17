from datetime import datetime, timedelta

import pyspark
import argparse

from utils import _update_impala_table


def execute_process(options):

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .appName("criar_tabela_andamento_processos") \
            .enableHiveSupport() \
            .getOrCreate()

    schema_exadata = options['schema_exadata']
    schema_exadata_aux = options['schema_exadata_aux']

    personagens_cutoff = options['personagens_cutoff']

    nb_past_days = options['nb_past_days']
    dt_inicio = datetime.now() - timedelta(nb_past_days)

    docu_totais = spark.sql(
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
        JOIN {1}.tb_regra_negocio_processo
            ON cod_pct = cod_atribuicao
            AND classe_documento = docu_cldc_dk
	    JOIN {0}.mcpr_classe_docto_mp ON cldc_dk = docu_cldc_dk
        WHERE PCAO_DT_ANDAMENTO >= '{2}'
        AND pcao_dt_cancelamento IS NULL
        AND docu_tpst_dk != 11
        """.format(schema_exadata, schema_exadata_aux, dt_inicio)
    )
    docu_totais.registerTempTable('docu_totais')

    personagens = spark.sql(
        """
        SELECT 
            docu_nr_mp,
            concat_ws(', ', collect_list(nm_personagem)) as personagens
        FROM (
            SELECT
                docu_nr_mp,
                CASE
                    WHEN nr_pers = {1} THEN 'e outros...'
                    ELSE pess_nm_pessoa END
                AS nm_personagem,
                nr_pers
            FROM (
                SELECT
                    docu_nr_mp,
                    pess_nm_pessoa,
                    row_number() OVER (PARTITION BY docu_nr_mp ORDER BY pess_dk DESC) as nr_pers
                FROM DOCU_TOTAIS
                JOIN {0}.mcpr_personagem ON pers_docu_dk = docu_dk
                AND pers_tppe_dk IN (290, 7, 21, 317, 20, 14, 32, 345, 40, 5)
                JOIN {0}.mcpr_pessoa ON pers_pess_dk = pess_dk) t
            WHERE nr_pers <= {1}) t1
        GROUP BY docu_nr_mp
        """.format(schema_exadata, personagens_cutoff)
    )
    personagens.registerTempTable('docu_personagens')

    dts_ultimos_andamentos = spark.sql(
        """
        SELECT
            DOCU_NR_MP,
            MAX(PCAO_DT_ANDAMENTO) AS DT_ULTIMO
        FROM DOCU_TOTAIS
        GROUP BY DOCU_NR_MP
        """
    )
    dts_ultimos_andamentos.registerTempTable('dts_ultimos_andamentos')


    lista_processos = spark.sql(
        """
        SELECT
            A.orgao_dk,
            A.classe_documento as cldc_dk,
            A.docu_nr_mp,
            A.docu_nr_externo,
            A.docu_tx_etiqueta,
            P.personagens,
            A.pcao_dt_andamento as dt_ultimo_andamento, 
            concat_ws(', ', collect_list(A.tppr_descricao)) as ultimo_andamento,
            CASE WHEN length(docu_nr_externo) = 20 THEN 
                concat('http://www4.tjrj.jus.br/numeracaoUnica/faces/index.jsp?numProcesso=',
                    concat(concat(concat(concat(concat(concat(
                    concat(substr(docu_nr_externo, 1, 7), '-')), 
                    concat(substr(docu_nr_externo, 8, 2), '.')),
                    concat(substr(docu_nr_externo, 10, 4), '.')),
                    concat(substr(docu_nr_externo, 14, 1), '.')),
                    concat(substr(docu_nr_externo, 15, 2), '.')),
                    substr(docu_nr_externo, 17, 4)))
                ELSE NULL
            END as url_tjrj
        FROM DOCU_TOTAIS A
        JOIN DTS_ULTIMOS_ANDAMENTOS ULT 
            ON A.DOCU_NR_MP = ULT.DOCU_NR_MP
            AND A.PCAO_DT_ANDAMENTO = ULT.DT_ULTIMO
        JOIN DOCU_PERSONAGENS P ON P.DOCU_NR_MP = A.DOCU_NR_MP
        GROUP BY A.orgao_dk, A.classe_documento, A.docu_nr_mp, 
            A.docu_nr_externo, A.docu_tx_etiqueta, P.personagens,
            A.pcao_dt_andamento
        """
    )

    table_name = "{}.tb_lista_processos".format(schema_exadata_aux)

    lista_processos.write.mode("overwrite").saveAsTable("temp_table_lista_processos")
    temp_table = spark.table("temp_table_lista_processos")

    temp_table.write.mode("overwrite").saveAsTable(table_name)
    spark.sql("drop table temp_table_lista_processos")

    _update_impala_table(table_name, options['impala_host'], options['impala_port'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create table distribuicao entradas")
    parser.add_argument('-e','--schemaExadata', metavar='schemaExadata', type=str, help='')
    parser.add_argument('-a','--schemaExadataAux', metavar='schemaExadataAux', type=str, help='')
    parser.add_argument('-i','--impalaHost', metavar='impalaHost', type=str, help='')
    parser.add_argument('-o','--impalaPort', metavar='impalaPort', type=str, help='')
    parser.add_argument('-c','--personagensCutoff', metavar='personagensCutoff', type=int, default=2, help='')
    parser.add_argument('-p','--nbPastDays', metavar='nbPastDays', type=int, default=7, help='')
    
    args = parser.parse_args()

    options = {
                    'schema_exadata': args.schemaExadata, 
                    'schema_exadata_aux': args.schemaExadataAux,
                    'impala_host' : args.impalaHost,
                    'impala_port' : args.impalaPort,
                    'personagens_cutoff' : args.personagensCutoff,
                    'nb_past_days': args.nbPastDays
                }

    execute_process(options)
