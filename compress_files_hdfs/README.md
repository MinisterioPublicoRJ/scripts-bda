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

## Archive
The files also are archived in BACKUP_STAGING directory HDFS in case of error it can be recovered. For unarchived use this command:

    hadoop distcp <src> <dst>
    
    Example:
    hadoop distcp  har:///user/example/test.har/*.gz hdfs:///user/example/unarchive_test
