'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import numpy as np
import pandas as pd
from pyspark.mllib.evaluation import RankingMetrics 
from pyspark.ml.evaluation import RegressionEvaluator


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    # read in test data
    # try with train data instead just for debugging
    #test = spark.read.csv(f'hdfs:/user/{netID}/ratings_small_train.csv', header='true', schema='index INT, userId INT,movieId INT,rating DOUBLE,timestamp INT')


    test = spark.read.csv(f'hdfs:/user/{netID}/ratings_small_test.csv', header='true', schema='index INT, userId INT,movieId INT,rating DOUBLE,timestamp INT')
    #test.show()    

    # get ground truth set for each user
    ground_truth_test = test.groupBy('userId').agg(collect_set('movieId').alias('ground_truth')).repartition('userId')


    # read in popularity predictions
    top100 = spark.read.csv(f'hdfs:/user/{netID}/top100_pop_small.csv', header='false', schema='movieId INT')

    # convert to numpy array
    top100_pd = top100.toPandas()
    top100_arr = top100_pd.to_numpy().flatten()
    #top100_arr = top100_pd.to_numpy()
    print(top100_arr)

    test_users = test.select('userId').distinct()    
    predictions = test_users.withColumn('prediction', array([lit(i) for i in top100_arr.tolist()]))    
   # predictions = test_users.withColumn('prediction', top100_arr)     
    
    #predictions.show()

    combo = predictions.join(broadcast(ground_truth_test), on = 'userId', how = 'inner')
    #combo.show()


    #predictionAndLabels = combo.rdd.map(lambda row: (row['movieId'], row['ground_truth']))


    predictionAndLabels = combo.rdd.map(lambda row: (row['prediction'], row['ground_truth']))
    
    #print(predictionAndLabels.take(3))

    metrics = RankingMetrics(predictionAndLabels)

    MAP = metrics.meanAveragePrecision
    print('MAP:',MAP) 
    precis = metrics.precisionAt(100)
    print("precis", precis)










# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('evalBaselinePop').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
