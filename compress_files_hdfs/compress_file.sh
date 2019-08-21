#!/bin/bash

function usage()
{
    echo -e " PROCESS TO COMPRESS FILES IN HDFS "
    echo -e " Usage: $0 [--help] <input-path> "
    echo -e " <input-path> : input path "
    echo -e " files types like {*.csv}, {*.pdf}, or {*.pdf,*.txt} etc. "
    echo -e "--help    : print help of the program."
    echo -e ""
    exit
}

function process()
{
    echo "Compactando os arquivos..."
    hadoop jar /opt/cloudera/parcels/CDH-5.14.2-1.cdh5.14.2.p0.3/lib/hadoop-mapreduce/hadoop-streaming-2.6.0-cdh5.14.2.jar \
        -Dmapreduce.output.fileoutputformat.compress=true \
        -Dmapreduce.map.output.compress=true \
        -Dmapreduce.output.fileoutputformat.compress.codec=org.apache.hadoop.io.compress.GzipCodec \
        -Dmapreduce.job.reduces=1 \
        -Dmapreduce.input.fileinputformat.split.minsize=2000000000 \
        -mapper "grep -v $header_line" \
        -reducer /bin/cat \
        -input $input_path \
        -output $output_path

    readarray -t output_files <<< "$(hdfs dfs -ls "$output_path"/*.gz |  tr -s ' ' | cut -d' ' -f8)"
    for file_out in "${output_files[@]}"
    do 
        data_atual=$(date +%Y%m%d%s)
        final_file=output_compressed_file_$data_atual.gz
        hdfs dfs -cp "$file_out" "$compressed_path"/$final_file
    done
}

function get_header()
{
    echo "Recuperando o header dos arquivos"
    file="$(hdfs dfs -ls $1 | head -2 | tr -s ' ' | cut -d' ' -f8)"
    header_line="$(hdfs dfs -text $file | head -n 1)"
    header_line=${header_line%$'\r'}
}

function move_files()
{
    echo "Movendo dados para a pasta temporaria"
    type_files=$1
    hdfs dfs -mv $path_dir/$type_files $temp_path

    data_atual=$(date +%Y%m%d%s)
    echo "Arquivando os arquivos originais"
    hadoop archive -archiveName archive.har -p $temp_path $path_dir/archive-$data_atual
}

input_file=$1

if [[ $# -eq 1 && "$1" == "--help" ]]; then
	usage
fi

# verifying the number of arguments used in execution of program...
if [[ $# != 1 ]]; then
	echo "Wrong number of arguments passed to the program..."
	usage
fi

# verifying if first argument is a regular file...
if [[ ! -f "$input_file" ]]; then
	echo "The second argument '"$input_file"' is not a regular file!"
	exit
fi

readarray list_dir < $input_file


for dir in "${list_dir[@]}"
do	
    dir=$(echo $dir | xargs)

    # Get only path without file name or extension
    path_dir=${dir%/*}
    # Get only extension file
    file_extension=$(basename "$dir" )


    hdfs dfs -test -d $path_dir
    if [[ $? == 0  ]]; then

        echo "Criando repositorio temporario"
        temp_path="$path_dir"/temp
        hdfs dfs -mkdir $temp_path

        get_header "$path_dir"

        echo "Criando repositorio final"
        compressed_path="$path_dir"/compressed
        hdfs dfs -mkdir $compressed_path

        if [[ $file_extension == '{*.gz}' || $file_extension == '*.gz' ]]; then

            move_files "$file_extension"

            echo "Extraindo os arquivos gz e convertendo em um arquivo csv"
            # Extract files and the result is output. Put the result in one file in hdfs
            hdfs dfs -text $temp_path/*.gz | hdfs dfs -put - $temp_path/output_file.csv
            
            input_path=$temp_path/*.csv
            output_path=$temp_path/output

        else
            
            move_files "$file_extension"
            input_path=$temp_path
            output_path=$temp_path/output

        fi

        process
        echo "Removendo os arquivos temporarios"
        hdfs dfs -rm -r -skipTrash $temp_path
    else
		echo "The directory ""$path_dir"" not exist in HDFS." >> error.log
    fi

done