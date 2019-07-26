## Install Requirements
To install the requirements, run the following commands:

    make clean
    make install

## Run Script
All the main commands are executed through use of the run_classifier.sh script. To train the model, run:

    sh run_classifier.sh train

For prediction:

    sh run_classifier.sh predict

To test the scripts using pytest:

    sh run_classifier.sh test

## Environment Variables

URL_ORACLE_SERVER : The Oracle server URL to use to connect to the database containing the data

USER_ORACLE : The user to use to connect to the database

PASSWD_ORACLE : The password to use to connect to the database

ORACLE_DRIVER_PATH : The path to the oracle jdbc.jar driver file

ROBOT_NAME : The robot name to use when registering results back to the database

ROBOT_NUMBER : The robot number to use when registering results back to the database

HDFS_URL : The HDFS URL where the model and results will be stored

HDFS_USER : The HDFS user

HDFS_MODEL_DIR : The directory on the HDFS where the model and results will be stored

UPDATE_TABLES : Optional. Whether the predict phase should update the original database or not.

UFED_DK : Optional. The numerical code for the State to be analyzed.

START_DATE : Optional. The starting date used for the training and/or predict phase. In YYYY-MM-DD format.

END_DATE : Optional. The end date used for the training and/or predict phase. In YYYY-MM-DD format.

## Model Structure

The models are saved in the given file structure on HDFS:

    path/
        /model_date1
            /model
            /results
            /evaluate
        /model_date2
            /model
            /results
            /evaluate

Where /model_date is the date on which the model has been trained, using a YearMonthDayHourMinuteSecond notation.

The /model directory contains the model components, such as the model itself, the multilabel binarizer, and vectorizer used, as well as the documents keys that were used for training.

The /results directory contains the results given by the 'predict' phase, in a .csv format.

The /evaluate directory holds the metrics calculated by the 'evaluate' phase.
