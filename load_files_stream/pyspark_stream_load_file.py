from __future__ import print_function

import sys

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

def tf(file, iterator):
    values = list(iterator)
    yield (file, values)

def mapIndex(index, iterator, filenames):
    values = list(iterator)
    yield (filenames[index], values)

def f(rdd):
    debug = rdd.toDebugString()

    print(debug)
    filenames = []

    lines = debug.split("\n")[2:]
    for l in lines:
        filenames.append(l.split()[1].split("/")[-1])

    print(filenames)
    
    if not rdd.isEmpty():
        #rdd = rdd.mapPartitionsWithIndex(lambda index, iterator: mapIndex(index, iterator, filenames))
        rdd = rdd.mapPartitionsWithSplit(lambda file, iterator: tf(filenames, iterator))
        f = rdd.collect()
        for j in f:
            print(j)
        
        #rdd = rdd.mapPartitionsWithSplit(lambda file, iterator: tf(file, iterator))
        #g = rdd.filter(lambda (x, y): x == "tmp").map(lambda (x, y): y)
        #f = rdd.filter(lambda x: x[0] != "tmp").map(lambda x: x[1]).collect()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: hdfs_wordcount.py <directory>", file=sys.stderr)
        sys.exit(-1)

    sc = SparkContext(appName="PythonStreamingHDFSWordCount")
    ssc = StreamingContext(sc, 10)

    sparkConf = SparkConf() \
                .set("spark.hive.mapred.supports.subdirectories","false") \
                .set("spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive","false") \

    spark = pyspark.sql.session.SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()

    lines = ssc.textFileStream(sys.argv[1])

    lines.foreachRDD(f)
    # counts = lines.flatMap(lambda line: line.split(" "))\
    #               .map(lambda x: (x, 1))\
    #               .reduceByKey(lambda a, b: a+b)

    #counts.pprint()

    ssc.start()
    ssc.awaitTermination()