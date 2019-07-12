To install the requirements, run the following commands:
    make clean
    make install

All the main commands are executed through use of the run_classifier.sh script. To train the model, run:
    sh run_classifier.sh train

For prediction:
    sh run_classifier.sh predict

To test the scripts using pytest:
    sh run_classifier.sh test

The following environment variables must be declared:

URL_ORACLE_SERVER : The Oracle server URL to use to connect to the database containing the data
USER_ORACLE : The user to use to connect to the database
PASSWD_ORACLE : The password to use to connect to the database
ORACLE_DRIVER_PATH : The path to the oracle jdbc.jar driver file
ROBOT_NAME : The robot name to use when registering results back to the database
ROBOT_NUMBER : The robot number to use when registering results back to the database
HDFS_URL : The HDFS URL where the model and results will be stored
HDFS_USER : The HDFS user
HDFS_MODEL_DIR : The directory on the HDFS where the model and results will be stored