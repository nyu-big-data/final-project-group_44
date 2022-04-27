'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
import numpy as np 
from pyspark.sql.functions import *
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession







def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    # load model
    model = ALSModel.load(f"hdfs:/user/ya2193/ALS_model_baseline_small")

    # load test data
    test = spark.read.csv(f'hdfs:/user/{netID}/ratings_small_test.csv', header='true', schema='index INT, userId INT,movieId INT,rating DOUBLE,timestamp INT')
    
    # get ground truth for each user
    test_users = test.select('userId').distinct()     
    ground_truth_test = test.groupBy('userId').agg(collect_set('movieId').alias('ground_truth')).repartition('userId')
    
    # get predictions for each user
    predictions = model.recommendForUserSubset(test_users, 100).select('userId', 'recommendations.movieId').repartition("userId")
    
    combo = predictions.join(broadcast(ground_truth_test), on = 'userId', how = 'inner')

    predictionAndLabels = combo.rdd.map(lambda row: (row['movieId'], row['ground_truth']))

    # Metrics
    metrics = RankingMetrics(predictionAndLabels)

    MAP = metrics.meanAveragePrecision
    print('MAP:',MAP) 
    precis = metrics.precisionAt(100)
    print("precis:", precis)
    NDCG = metrics.ndcgAt(100)
    print('NDCG:', NDCG)




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
