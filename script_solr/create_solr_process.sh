#!/bin/bash

function usage()
{
  echo -e " PROCESS TO LOAD ENTITYS IN SOLR SERVER "
  echo -e " Usage: $0 [--help]  <server> <base-path> <type-process> <name-entity>"
  echo -e " <server> : Server http where solr is "
  echo -e " <base-path> : Base path where the entity configuration is "
  echo -e " <type-process> :  type of the process to be executed:   "
  echo -e "                   1 - For create and load full data for all entitys "
  echo -e "                   2 - For create and load full data for one entity "
  echo -e "                   3 - Update the entity and load full data "
  echo -e "                   4 - Load only new data of the informed entity  "
  echo -e " <name-entity> : Entity's name "
  echo -e "--help    : print help of the program."
  echo -e ""
  exit
}

SERVER=$1
BASE_PATH=$2
TYPE_PROCESS=$3
NAME_ENTITY=$4
OPTIONS=(1 2 3 4)


ARRAY_ENTITYS=(documento_personagem embarcacao veiculo pessoa_juridica pessoa_fisica)

if [[ $# -lt 3 || "$1" == "--help" ]]; then
	usage
fi

# verifying the number of arguments used in execution of program...
if [[ $# -gt 4 ]]; then
	echo "Wrong number of arguments passed to the program..."
	usage
fi

if [[ ! " ${OPTIONS[@]} " =~ " $TYPE_PROCESS " ]]; then 
    echo "There is not option $TYPE_PROCESS"
    usage
fi


if [[ "$TYPE_PROCESS" -eq 1 ]]; then
    for entity in "${ARRAY_ENTITYS[@]}"
    do  
        solrctl collection --delete $entity
        solrctl instancedir --delete $entity

        solrctl instancedir --create $entity $BASE_PATH$entity
        solrctl collection --create $entity -s 3 -m 3

        curl "$SERVER/solr/$entity/dataimport?command=full-import&clean=true&entity=$entity"

        if [ $? -eq 1 ]; then  
            echo "Error create entity in solr"
            usage
        fi
   done
fi


if [ -z "$3" ] && [ "$TYPE_PROCESS" != "1" ]; then
    echo "Argument of entity name is empty"
	usage
fi

if [ "$TYPE_PROCESS" -eq "2" ]; then

    solrctl collection --delete $NAME_ENTITY
    solrctl instancedir --delete $NAME_ENTITY

    solrctl instancedir --create $NAME_ENTITY $BASE_PATH$NAME_ENTITY
    solrctl collection --create $NAME_ENTITY -s 3 -m 3

    curl "$SERVER/solr/$NAME_ENTITY/dataimport?command=full-import&clean=true&entity=$NAME_ENTITY"

fi

if [ "$TYPE_PROCESS" -eq "3" ]; then

    solrctl instancedir --update $NAME_ENTITY $BASE_PATH$NAME_ENTITY
    solrctl collection --delete $NAME_ENTITY
    solrctl collection --create $NAME_ENTITY -s 3 -m 3
    curl "$SERVER/solr/$NAME_ENTITY/dataimport?command=full-import&clean=true&entity=$NAME_ENTITY"

fi

if [ "$TYPE_PROCESS" -eq "4" ]; then
    curl "$SERVER/solr/$NAME_ENTITY/dataimport?command=full-import&clean=false&entity=$NAME_ENTITY"
fi

