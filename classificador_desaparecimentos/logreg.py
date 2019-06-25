from base import spark

import pandas as pd
import numpy as np
import re

from copy import deepcopy
from unidecode import unidecode
from decouple import config

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

rules = {
    2: ['USUARI[OA] DE (DROGA|ENTORPECENTE)S?', 'ALCOOLATRA', 'COCAINA', 'VICIAD[OA]', 'DEPENDENTE QUIMICO', 'MACONHA', 'ALCOOL', 'CRACK'],
    5: ['DEPRESSAO', 'ESQUI[ZS]OFRENIA', 'ESQUI[ZS]OFRENIC[OA]', 'ALZHEIMER', 
                                '(DOENCA|TRANSTORNO|PROBLEMA|DISTURBIO)S? MENTA(L|IS)'],
    4: [' TRAFICO', 'TRAFICANTES'],
    20: ['ABORDAD[OA] (POR POLICIAIS|PELA POLICIA)'],
    3: ['FORTES CHUVAS', 'TEMPESTADE', 'ENXURRADA', 'DESLIZAMENTO', 'ROMPIMENTO D[EA] BARRAGEM', 'SOTERRAD[OA]'],
    6: [' RAPTOU ', ' RAPTAD[OA] ', 'SEQUESTROU?', 'SEQUESTRAD[OA]']
}

URL_ORACLE_SERVER = config('URL_ORACLE_SERVER')
USER_ORACLE = config('USER_ORACLE')
PASSWD_ORACLE = config('PASSWD_ORACLE')

# TODO: define variables for label column name, text name, etc

query = """
(SELECT A.SNCA_DK, B.UFED_SIGLA, A.SNCA_IDENTIFICADOR_SINALID, A.SNCA_DS_FATO, 
A.SNCA_DT_FATO, A.SNCA_PC_RELEVANCIA_POR_DOCTO, A.SNCA_IN_TRAFICO_PESSOAS,
D.MDEC_DK, D.MDEC_DS_MOTIVO_OCORRENCIA, D.MDEC_IN_SOMENTE_INSTAURACAO, E.SISI_SITUACAO_SINDICANCIA
FROM SILD.SILD_SINDICANCIA A
INNER JOIN CORP.CORP_UF B ON B.UFED_DK = A.SNCA_UFED_DK
INNER JOIN SILD.SILD_DESAPARE_MOT_DECLARADO C ON C.DMDE_SDES_DK = A.SNCA_DK
INNER JOIN SILD.SILD_MOTIVO_DECLARADO D ON D.MDEC_DK = C.DMDE_MDEC_DK
INNER JOIN SILD.SILD_SITUACAO_SINDICANCIA E ON E.SISI_DK = A.SNCA_SISI_DK) t 
"""

# --UPDATE SITUACAO SINDICANCIA (FINALIZADA = TREINO, A INSTAURAR = A PREDIZER)

data = spark.read.format("jdbc") \
.option("url", URL_ORACLE_SERVER) \
.option("dbtable", query) \
.option("user", USER_ORACLE) \
.option("password", PASSWD_ORACLE) \
.option("driver", "oracle.jdbc.driver.OracleDriver") \
.load()

clean_text = udf(lambda x: re.sub('[^a-zA-Z ]', '', unidecode(x).upper()), StringType())

data = data.na.drop(subset=["SNCA_DS_FATO"])
data = data.withColumn("SNCA_DS_FATO", clean_text(data.SNCA_DS_FATO))
data = data.withColumn("MDEC_DK", data.MDEC_DK.cast(IntegerType()))
data = data.withColumn("SNCA_DK", data.SNCA_DK.cast(IntegerType()))

labels = [r.MDEC_DK for r in data.select('MDEC_DK').distinct().collect()]

# TODO: Train and test should take into considered SITUACAO_SINDICANCIA
trainData = data.filter("UFED_SIGLA != 'SP' AND UFED_SIGLA != 'CE'").select("SNCA_DK" ,"MDEC_DK", "SNCA_DS_FATO")
testData = data.filter("UFED_SIGLA = 'SP' OR UFED_SIGLA = 'CE'").select("SNCA_DK" ,"MDEC_DK", "SNCA_DS_FATO")

columns = trainData.columns
columns.remove('MDEC_DK')

trainData = trainData.groupBy(columns).agg(collect_set('MDEC_DK').alias('MDEC_DK'))

tokenizer = Tokenizer(inputCol="SNCA_DS_FATO", outputCol="words")
wordsData = tokenizer.transform(trainData)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2000)
featurizedData = hashingTF.transform(wordsData)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData = rescaledData.withColumnRenamed('MDEC_DK', 'label')

has_class_i = udf(lambda x, i: 1 if i in x else 0, IntegerType())

models = []
for i in labels:
    ith_dataset = rescaledData.withColumn('label', has_class_i(rescaledData.label, lit(i)))
    ith_dataset.cache()
    # Define Logistic Regression model
    lr = LogisticRegression(maxIter=10)
    # Fit the model
    lrModel = lr.fit(ith_dataset)
    models.append(lrModel)
    # Print the coefficients and intercept for logistic regression
    print("Coefficients for class {}: {}".format(i, lrModel.coefficients))
    print("Intercept for class {}: {}".format(i, lrModel.intercept))

#Test prediction
testSentences = testData.select('SNCA_DK', 'SNCA_DS_FATO')

replace_class_number = udf(lambda x, i: i if x == 1 else None)

#Regex Classifier
def regex_classify(text):
    results = []
    for c in rules:
        for expression in rules[c]:
            m = re.search(expression, unidecode(text).upper())
            if m:
                results.append(c)
                break
    return results
udf_regex_classify = udf(regex_classify)

predictions_regex = []
regexPredictions = testData.withColumn('regex_predictions', udf_regex_classify(testData.SNCA_DS_FATO))

testData = tokenizer.transform(testData)
testData = hashingTF.transform(testData)
testData = idfModel.transform(testData)

predictions_DFs = []
for i, ith_model in zip(labels, models):
    predictions = ith_model.transform(testData).select('SNCA_DK', 'prediction')
    predictions = predictions.withColumnRenamed('SNCA_DK', 'id')
    predictions = predictions.withColumn('prediction_{}'.format(i), replace_class_number(predictions.prediction, lit(i)))
    predictions.registerTempTable('prediction_{}'.format(i))
    predictions_DFs.append(predictions)

_schema = copy.deepcopy(testSentences.schema)
predicted_test = testSentences.rdd.toDF(_schema)
predicted_test = predicted_test.withColumnRenamed('SNCA_DK', 'id_sentence')
predicted_test.registerTempTable('testTable')

# TODO: automatize query generation based on number of classes and their values
predicted_test = spark.sql("""
    SELECT A.id_sentence, A.SNCA_DS_FATO, 
    array(B.prediction_1, 
    C.prediction_2, 
    D.prediction_3, 
    E.prediction_4,
    F.prediction_5,
    G.prediction_6,
    H.prediction_7,
    I.prediction_8,
    J.prediction_9,
    K.prediction_10,
    L.prediction_11,
    M.prediction_12,
    N.prediction_13,
    O.prediction_15,
    P.prediction_20) as predictions
    FROM testTable A
    INNER JOIN prediction_1 B ON B.id = A.id_sentence
    INNER JOIN prediction_2 C ON C.id = A.id_sentence
    INNER JOIN prediction_3 D ON D.id = A.id_sentence
    INNER JOIN prediction_4 E ON E.id = A.id_sentence
    INNER JOIN prediction_5 F ON F.id = A.id_sentence
    INNER JOIN prediction_6 G ON G.id = A.id_sentence
    INNER JOIN prediction_7 H ON H.id = A.id_sentence
    INNER JOIN prediction_8 I ON I.id = A.id_sentence
    INNER JOIN prediction_9 J ON J.id = A.id_sentence
    INNER JOIN prediction_10 K ON K.id = A.id_sentence
    INNER JOIN prediction_11 L ON L.id = A.id_sentence
    INNER JOIN prediction_12 M ON M.id = A.id_sentence
    INNER JOIN prediction_13 N ON N.id = A.id_sentence
    INNER JOIN prediction_15 O ON O.id = A.id_sentence
    INNER JOIN prediction_20 P ON P.id = A.id_sentence
""")

format_log_predictions = udf(lambda x: [int(p) for p in x if p is not None])

predicted_test = predicted_test.withColumn('predictions', format_log_predictions(predicted_test.predictions))

regex_else_logreg = udf(lambda a, b: a if a != '[]' else b)
predicted_test.registerTempTable('log_reg_predictions')
regexPredictions.registerTempTable('regex_predictions')

predicted_test = spark.sql("""
    SELECT A.id_sentence, A.SNCA_DS_FATO,
    B.regex_predictions, A.predictions
    FROM log_reg_predictions A
    INNER JOIN regex_predictions B ON B.SNCA_DK = A.id_sentence
""")

predicted_test = predicted_test.withColumn('predictions', regex_else_logreg(predicted_test.regex_predictions, predicted_test.predictions))
predicted_test = predicted_test.select("id_sentence", "SNCA_DS_FATO", "predictions")

clean_prediction_string = udf(lambda x: re.sub('[\[\]]', '', x))

predicted_test = predicted_test.withColumn('predictions', clean_prediction_string(predicted_test.predictions))
predicted_test = predicted_test.withColumn('predictions', explode(split(predicted_test.predictions, ',')))
predicted_test.show()

# clean_pred_test_data = spark.createDataFrame([
#     ('[1,5]', "Hi I heard about Spark"),
#     ('[13]', "I wish Java could use case classes"),
#     ('[1,4,9]', "Logistic regression models are neat")
# ], ["label", "sentence"])

# clean_pred_test_data.show()
# clean_pred_test_data = clean_pred_test_data.withColumn('label', clean_prediction_string(clean_pred_test_data.label))
# clean_pred_test_data = clean_pred_test_data.withColumn('label', explode(split(clean_pred_test_data.label, ',')))
# clean_pred_test_data.show()