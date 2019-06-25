echo $1
direc=$1
cd $direc
python setup.py bdist_egg
mv dist/*.egg ../