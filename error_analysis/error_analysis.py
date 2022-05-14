'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass
import builtins

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
import numpy as np
from sklearn.metrics import average_precision_score as AP

from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.types import FloatType


def apk(predicted, actual, k=100):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0
    
    return score / builtins.min([len(actual), k])




def main(spark, netID):
	'''Main routine for Lab Solutions
	Parameters
	----------
	spark : SparkSession object
	netID : string, netID of student to find files in HDFS
	'''

	# load best model
	model = ALSModel.load(f"hdfs:/user/{netID}/ALS_model_small_rank30_reg0.1")
	# load test data
	test = spark.read.csv(f'hdfs:/user/{netID}/ratings_small_test.csv', header='true', schema='index INT, userId INT,movieId INT,rating DOUBLE,timestamp INT')
	# get ground truth for each user
	test_users = test.select('userId').distinct()
	ground_truth_test = test.groupBy('userId').agg(collect_set('movieId').alias('ground_truth')).repartition('userId')

	# get predictions for each user
	predictions = model.recommendForUserSubset(test_users, 100).select('userId', 'recommendations.movieId').repartition("userId")

	combo = predictions.join(broadcast(ground_truth_test), on = 'userId', how = 'inner')

	predictAndTruth = combo.select('movieId', 'ground_truth')

	# create udf of our AP function
	ap_udf = udf(apk, FloatType())


	map_vals = predictAndTruth.withColumn('AP', ap_udf('movieId', 'ground_truth'))

	map_vals.show()




# Only enter this block if we're in main
if __name__ == "__main__":

        # Create the spark session object
        spark = SparkSession.builder.appName('part1').getOrCreate()

        # Get user netID from the command line
        netID = getpass.getuser()

        # Call our main routine
        main(spark, netID)
