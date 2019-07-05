# -*- coding: utf-8 -*-
import pandas as pd

def get_train_data(cursor, UFED_DK=None):
    TRAIN_QUERY = """
        SELECT DISTINCT B.SNCA_DK, B.SNCA_DS_FATO, D.DMDE_MDEC_DK
        FROM SILD.SILD_ATIVIDADE_SINDICANCIA A
        INNER JOIN SILD.SILD_SINDICANCIA B 
            ON A.ATSD_SNCA_DK = B.SNCA_DK
        INNER JOIN SILD.SILD_DESAPARE_MOT_DECLARADO D 
            ON D.DMDE_SDES_DK = B.SNCA_DK
        WHERE A.ATSD_TPSN_DK = 22
    """
    if UFED_DK:
        TRAIN_QUERY += " AND B.SNCA_UFED_DK = {}".format(UFED_DK)

    cursor.execute(TRAIN_QUERY)

    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=columns)


def get_predict_data(cursor, UFED_DK=None):   
    PREDICT_QUERY = """
        SELECT DISTINCT B.SNCA_DK, B.SNCA_DS_FATO, D.DMDE_MDEC_DK
        FROM SILD.SILD_ATIVIDADE_SINDICANCIA A
        INNER JOIN SILD.SILD_SINDICANCIA B
            ON A.ATSD_SNCA_DK = B.SNCA_DK
        INNER JOIN SILD.SILD_DESAPARE_MOT_DECLARADO D
            ON D.DMDE_SDES_DK = B.SNCA_DK
        WHERE A.ATSD_TPSN_DK = 2
        AND NOT EXISTS (
            SELECT ATSD_SNCA_DK
            FROM SILD.SILD_ATIVIDADE_SINDICANCIA B 
            WHERE B.ATSD_SNCA_DK = A.ATSD_SNCA_DK AND B.ATSD_TPSN_DK = 22)
    """
    if UFED_DK:
        PREDICT_QUERY += " AND B.SNCA_UFED_DK = {}".format(UFED_DK)

    cursor.execute(PREDICT_QUERY)

    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=columns)


def set_module_and_client(cursor, client_name):
    cursor.execute("CALL dbms_application_info.set_module(?, ?)",
                   ('SILD', 'Funcionalidade'))
    cursor.execute("CALL dbms_application_info.set_client_info(?)",
                   (client_name,))

def get_max_dk(cursor, table_name, column_name):
    cursor.execute("SELECT MAX({}) FROM {}".format(column_name, table_name))
    return int(cursor.fetchall()[0][0])

def update_atividade_sindicancia(cursor, ativ_dk, snca_dk, 
                                 user_name, user_number):
    ATIV_SINDICANCIA_QUERY = """
        INSERT INTO SILD.SILD_ATIVIDADE_SINDICANCIA
        (ATSD_DK, ATSD_SNCA_DK, ATSD_TPSN_DK, ATSD_DT_REGISTRO, 
        ATSD_DS_MOTIVO_ATIVIDADE, ATSD_NM_RESP_CTRL, ATSD_CPF_RESP_CTRL)
        VALUES (?, ?, 23, SYSDATE, 'CLASSIFICACAO FEITA PELO ROBO', ?, ?)
    """

    cursor.execute(ATIV_SINDICANCIA_QUERY,
                   (int(ativ_dk), int(snca_dk), user_name, user_number))

def update_motivo_declarado(cursor, snca_dk, labels):
    DELETE_MOT_DECLARADO_QUERY = """
        DELETE FROM SILD.SILD_DESAPARE_MOT_DECLARADO WHERE DMDE_SDES_DK = ?
    """

    INSERT_MOT_DECLARADO_QUERY = """
        INSERT INTO SILD.SILD_DESAPARE_MOT_DECLARADO 
        (DMDE_SDES_DK, DMDE_MDEC_DK) 
        VALUES (?, ?)
    """
    cursor.execute(DELETE_MOT_DECLARADO_QUERY, 
                   (int(snca_dk),))
    for label in labels: 
        cursor.execute(INSERT_MOT_DECLARADO_QUERY, 
                       (int(snca_dk), int(label)))