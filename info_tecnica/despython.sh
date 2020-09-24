echo $1
direc=$1
cd $direc
/opt/cloudera/parcels/Anaconda-5.0.1/bin/python setup.py bdist_egg
if [[ $? == 0 ]]; then
    mv dist/*.egg ../
else
    folder_name=$(basename "$direc" )
    name_zip=(${folder_name//-/ })
    cd $name_zip
    if [[ $? != 0 ]]; then
        cd src
	zip -r "$name_zip.zip" *
	mv "$name_zip.zip" ../../
    fi
fi
