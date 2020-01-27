echo $1
direc=$1
cd $direc
python setup.py bdist_egg
if [[ $? == 0 ]]; then
    mv dist/*.egg ../
else
    cd ..
    folder_name=$(basename "$direc" )
    zip -r "$folder_name.zip" $direc
fi
