import os
import argparse

from hdfs.ext.kerberos import KerberosClient
from base import spark_session
from happybase_kerberos_patch import KerberosConnection

""" 
    Connection to hbase
"""
def get_connection(server):
    connection = KerberosConnection(
            server, 
            protocol='compact', 
            use_kerberos=True, 
            timeout=300000
        )
    return connection

""" 
    Get table hbase
"""
def get_table(table_name, server):
    try:
        return get_connection(server).table(table_name)
    except:
        return get_connection(server).table(table_name)

"""
    Method to save byte files extrated to HDFS
"""
def save_file_hdfs(rdd, dir_files_pdf, server_hdfs, user_name_hdfs):
    n_file_id = int(rdd[0])
    n_info_tec = rdd[1].replace("/", "-")
    n_file = rdd[2]
    hdfsclient = KerberosClient(server_hdfs)
    hdfsclient.write(os.path.join(dir_files_pdf, '{}_{}.pdf'.format(n_file_id, n_info_tec)), n_file, overwrite=True)
    return rdd

def execute_process(args):

    spark = spark_session("info_tecnica")

    url_oracle_server = args.jdbcServer
    user_oracle = args.jdbcUser
    passwd_oracle = args.jdbcPassword

    dir_files_pdf = args.dirFilesPdf
    server_hdfs = args.hdfsServer
    user_name_hdfs = args.hdfsUser

    server = args.hbaseServer
    
    table_oracle = "GATE.GATE_INFO_TECNICA"

    row = get_table('file_info_tecnica', server).row('row1')
    last_id = int(row["last_id:"])


    #Connect to oracle to get all table GATE.GATE_INFO_TECNICA to extract files
    table = spark.read.format("jdbc").option("url", url_oracle_server) \
    .option("dbtable", table_oracle) \
    .option("user", user_oracle) \
    .option("password", passwd_oracle) \
    .option("driver", "oracle.jdbc.driver.OracleDriver") \
    .load()

    #Get id and byte file from rows greater than last_id variable
    select_file = table.where("ITCN_DK > {} and ITCN_ARQUIVO is not null".format(last_id)).select("ITCN_DK", "ITCN_NR_INFOTECNICA", "ITCN_ARQUIVO")

    #Transform in RDD and call the method save to HDFS
    select_file.rdd.map(lambda rdd: save_file_hdfs(rdd, dir_files_pdf, server_hdfs, user_name_hdfs)).count()

    #Get max id value from table oracle
    max_value = table.groupBy().max("ITCN_DK").first()[0]

    #Update value last_id in hbase table
    get_table('file_info_tecnica', server).put(b'row1',{b'last_id:': str(int(max_value))})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Save pdfs in hdfs")
    parser.add_argument('-s','--jdbcServer', metavar='jdbcServer', type=str, help='')
    parser.add_argument('-u','--jdbcUser', metavar='jdbcUser', type=str, help='')
    parser.add_argument('-p','--jdbcPassword', metavar='jdbcPassword', type=str, help='')
    parser.add_argument('-dfp','--dirFilesPdf', metavar='dirFilesPdf', type=str, help='')
    parser.add_argument('-hbs','--hbaseServer', metavar='hbaseServer', type=str, help='')
    parser.add_argument('-hs','--hdfsServer', metavar='hdfsServer', type=str, help='')
    parser.add_argument('-hu','--hdfsUser', metavar='hdfsUser', type=str, help='')
    args = parser.parse_args()

    execute_process(args)