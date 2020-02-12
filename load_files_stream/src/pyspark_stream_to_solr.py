from __future__ import print_function
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from decouple import config
import hashlib
import datetime
import sys
import pysolr

zookeeper_server = config('ZOOKEEPER_SERVER')
folder_hdfs = config("FOLDER_HDFS")
checkpoint_hdfs = config("CHECKPOINT_HDFS")

def getSparkSessionInstance(sparkConf):
    if ("sparkSessionSingletonInstance" not in globals()):
        globals()["sparkSessionSingletonInstance"] = pyspark.sql.session.SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]

def format_date(dt):
    return dt.replace(" ", "T") + "Z"

def connect_to_solr():
    zookeeper = pysolr.ZooKeeper(zookeeper_server)
    return pysolr.SolrCloud(zookeeper, "files_detran", timeout=300)

def send_data_to_solr(row):
    solr = connect_to_solr()
    solr.add([row], overwrite=True)

def generate_uuid(row):
    return hashlib.sha1(reduce(lambda x, y: x+ y, row)).hexdigest()

def check_row(row, header_list):
    data = dict(zip(tuple(header_list), row.split(',')))
    data['data'] = format_date(data['data'])
    data['uuid'] = generate_uuid(row.split(','))
    send_data_to_solr(data)
    return data

def verify_rdd(rdd, folder):

    if not rdd.isEmpty():

        spark = getSparkSessionInstance(rdd.context.getConf())

        header = 'num_camera,placa,lat,long,data,velocidade,faixa'
        header_list = header.split(',')
        rdd.filter(lambda r: r.lower() not in header).map(lambda w: check_row(w, header_list)).count()


def functionToCreateContext(folder):
    sparkConf = SparkConf()

    sparkConf.set("spark.dynamicAllocation.enabled", "false")
    sparkConf.set("spark.streaming.receiver.writeAheadLog.enable", "true")

    sc = SparkContext(appName="SPARK_STREAM_DETRAN_PLACAS",conf=sparkConf)

    sparkStreamingContext = StreamingContext(sc, 60)
    directoryStream = sparkStreamingContext.textFileStream("{}{}".format(folder_hdfs, folder))
    directoryStream.foreachRDD(lambda rdd: verify_rdd(rdd, folder))
    sparkStreamingContext.checkpoint(checkpoint)

    return sparkStreamingContext


if __name__ == "__main__":

    folder = sys.argv[1]

    checkpoint = '{}{}/'.format(checkpoint_hdfs, folder)

    ssc = StreamingContext.getOrCreate(checkpoint, lambda: functionToCreateContext(folder))
    ssc.start()
    ssc.awaitTermination()
    ssc.stop(False, False)
