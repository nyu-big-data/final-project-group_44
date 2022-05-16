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
from pyspark.sql.window import Window


def main(spark, netID):
	'''Main routine for Lab Solutions
	Parameters
	----------
	spark : SparkSession object
	netID : string, netID of student to find files in HDFS
	'''

	# load test data
	test = spark.read.csv(f'hdfs:/user/{netID}/pub/ratings_all_test.csv', header='true', schema='index INT, userId INT,movieId INT,rating DOUBLE,timestamp INT')
	#test = spark.read.csv(f'hdfs:/user/{netID}/ratings_small_test.csv', header='true', schema='index INT, userId INT,movieId INT,rating DOUBLE,timestamp INT')
	
	# get movies only
	test = test.select('movieId')

	# read in movie data
	movies = spark.read.csv(f'hdfs:/user/{netID}/movies_small.csv', header='true', schema='movieId INT, title STRING ,genres STRING')
	#movies = spark.read.csv(f'hdfs:/user/{netID}/movies_small.csv', header='true', schema='movieId INT, title STRING ,genres STRING')

	# get corresponding genres
	genres = test.join(movies, on='movieId', how='left').select('movieId', 'genres')
	genres.write.csv('test_genres_all.csv')
	#genres.write.csv('test_genres_small.csv')





# Only enter this block if we're in main
if __name__ == "__main__":

        # Create the spark session object
        spark = SparkSession.builder.appName('part1').getOrCreate()

        # Get user netID from the command line
        netID = getpass.getuser()

        # Call our main routine
        main(spark, netID)




