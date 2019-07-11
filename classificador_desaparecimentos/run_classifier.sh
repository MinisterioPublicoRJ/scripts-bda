#!/bin/sh
export PYTHONIOENCODING=utf8

#Usage statements
if [ "$#" = 0 ]; then
    echo "Usage: $0 command"
    echo "command must be either \"predict\", \"train\" or \"test\""
    exit 1
fi

command="${1}"

if [ ${command} = 'test' ]; then
    /opt/cloudera/parcels/Anaconda-5.0.1/bin/python src/run_tests.py
else
    spark2-submit --py-files packages/*.whl,packages/*.egg \
        --jars /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/jars/ojdbc6.jar \
        src/${command}.py 2>> error.log
fi