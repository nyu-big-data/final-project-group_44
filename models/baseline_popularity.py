'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    # Read in training data
    train = spark.read.csv(f'hdfs:/user/{netID}/ratings_small_train.csv', header='true', schema='index INT, userId INT,movieId INT,rating DOUBLE,timestamp INT')
    
    # make temporary view
    train.createOrReplaceTempView('train')

    # drop movies with 4 or less reviews
    train = spark.sql('SELECT *, COUNT(userId) OVER (PARTITION BY movieID) as num_ratings FROM train')
    train.createOrReplaceTempView('train')


    # get top 100 most popular movies
    top100 = spark.sql("SELECT movieId FROM train WHERE num_ratings > 4 GROUP BY movieId ORDER BY avg(rating) DESC LIMIT 100")
    #top100 = spark.sql("SELECT movieId FROM train GROUP BY movieId ORDER BY avg(rating) DESC LIMIT 100")

    top100.show()

    top100.coalesce(1).write.csv("top100_pop_small.csv")

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
