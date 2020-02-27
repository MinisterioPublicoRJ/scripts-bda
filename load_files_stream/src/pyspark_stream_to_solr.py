from __future__ import print_function
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from decouple import config
import hashlib
import datetime
import sys
import pysolr
import argparse

# zookeeper_server = config('ZOOKEEPER_SERVER')
# folder_hdfs = config("FOLDER_HDFS")
# checkpoint_hdfs = config("CHECKPOINT_HDFS")

def getSparkSessionInstance(sparkConf):
    if ("sparkSessionSingletonInstance" not in globals()):
        globals()["sparkSessionSingletonInstance"] = pyspark.sql.session.SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]

def format_date(dt):
    return dt.replace(" ", "T") + "Z"

def connect_to_solr(zookeeper_server):
    zookeeper = pysolr.ZooKeeper(zookeeper_server)
    return pysolr.SolrCloud(zookeeper, "files_detran", timeout=300)

def send_data_to_solr(row, zookeeper):
    solr = connect_to_solr(zookeeper)
    solr.add([row], overwrite=True)

def generate_uuid(row):
    return hashlib.sha1(reduce(lambda x, y: x+ y, row)).hexdigest()

def check_row(row, header_list, zookeeper):
    data = dict(zip(tuple(header_list), row.split(',')))
    data['datapassagem'] = format_date(data['datapassagem'])
    data['uuid'] = generate_uuid(row.split(','))
    send_data_to_solr(data, zookeeper)
    return data

def verify_rdd(rdd, zookeeper):

    if not rdd.isEmpty():

        spark = getSparkSessionInstance(rdd.context.getConf())

        header = 'num_camera,placa,lat,long,data,velocidade,faixa'
        header_list = ['num_camera','placa','lat','lon','datapassagem','velocidade','faixa']
        rdd.filter(lambda r: r.lower() not in header).map(lambda w: check_row(w, header_list, zookeeper)).count()


def functionToCreateContext(path, folder, zookeeper):
    sparkConf = SparkConf()

    sparkConf.set("spark.dynamicAllocation.enabled", "false")
    sparkConf.set("spark.streaming.receiver.writeAheadLog.enable", "true")

    sc = SparkContext(appName="SPARK_STREAM_DETRAN_PLACAS",conf=sparkConf)

    sparkStreamingContext = StreamingContext(sc, 60)
    directoryStream = sparkStreamingContext.textFileStream("{}{}".format(path, folder))
    directoryStream.foreachRDD(lambda rdd: verify_rdd(rdd, zookeeper))
    sparkStreamingContext.checkpoint(checkpoint)

    return sparkStreamingContext


if __name__ == "__main__":

    #folder = sys.argv[1]

    parser = argparse.ArgumentParser(description="Execute process stream Placas Detran")
    parser.add_argument('-f','--folderHDFS', metavar='folderHDFS', type=str, help='')
    parser.add_argument('-z','--zookeeperServer', metavar='zookeeperServer', type=str, help='')
    parser.add_argument('-p','--pathHDFS', metavar='pathHDFS', type=str, help='')
    parser.add_argument('-c','--checkpointHDFS', metavar='checkpointHDFS', type=str, help='')
    args = parser.parse_args()

    if args.folderHDFS and args.zookeeperServer and args.pathHDFS and args.checkpointHDFS:

        checkpoint = '{}{}/'.format(args.checkpointHDFS, args.folderHDFS)

        ssc = StreamingContext.getOrCreate(checkpoint, lambda: functionToCreateContext(args.pathHDFS, args.folderHDFS, args.zookeeperServer))
        ssc.start()
        ssc.awaitTermination()
        #ssc.awaitTerminationOrTimeout(36000)
        ssc.stop(False, False)

    else:
        parser.print_help()
        sys.exit(-1)
