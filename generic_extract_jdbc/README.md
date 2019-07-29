## Install Requirements
To install the requirements, run the following commands:

    make clean
    make install

## Run Script
All the main commands are executed through use of the generic_load_table.sh script:

    make

## Environment Variables

ORACLE_SERVER : The Oracle server URL to use to connect to the database containing the data

ORACLE_USER : The user to use to connect to the database

ORACLE_PASSWORD : The password to use to connect to the database

LOAD_ALL : Boolean value for load application. Use True for load all data or False to load just new data

IMPALA_HOST : The URL for host impala

IMPALA_PORT : Port for host impala
