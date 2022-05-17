'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass
import time

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
import numpy as np
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

     # Read in training data
    train = spark.read.csv(f'hdfs:/user/{netID}/ratings_small_train.csv', header='true', schema='index INT, userId INT,movieId INT,rating DOUBLE,timestamp INT')
    

    ranks = [30]
    regs = [0.1]

    for rank in ranks:
        for reg in regs:
            als = ALS(rank = rank, maxIter=20, regParam=reg, userCol="userId", itemCol="movieId", ratingCol="rating",\
                        nonnegative = True, implicitPrefs = True, coldStartStrategy="drop", seed=42)

            start = time.time()
            model = als.fit(train) 
            stop = time.time()
            model.write().overwrite().save(f"hdfs:/user/{netID}/ALS_model_small_rank{rank}_reg{reg}")
            print('time to run spark als on small set:', stop - start)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
