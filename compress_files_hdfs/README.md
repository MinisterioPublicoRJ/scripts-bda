#  Script to compress any directory file in HDFS
Script repository to compress any directories files with same shape and type.

## Run Script
The command are executed through use of the compress_file.sh script:

    ./compress_file.sh directories_hdfs.txt

## Directories file
The directories_hdfs file contains all the hdfs directories that want to be compressed. Ex:


    /user/staging/test1/{*.gz} or *.gz or *.csv
    /user/staging/test2/{*.gz}
    /user/staging/test3/{*.gz}
