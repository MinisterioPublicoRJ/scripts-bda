echo $1
direc=$1
cd $direc
/opt/cloudera/parcels/Anaconda-5.0.1/bin/python setup.py bdist_egg
