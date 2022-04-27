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

    ratings = spark.read.csv(f'hdfs:/user/{netID}/ratings.csv', header='true', schema='userId INT,movieId INT,rating DOUBLE,timestamp INT') # read in ratings for full dataset
   
    # create temp view
    ratings.createOrReplaceTempView('ratings')

    # create dataframe of userId ratings counts
    countsDF = spark.sql("SELECT userId, count(userId) as rating_counts FROM ratings GROUP BY userId") 
    
    # create temp view of countsDF
    countsDF.createOrReplaceTempView('counts')

    # merge to create new DF that includes counts
    to_split = spark.sql("SELECT r.userId, r.movieId, r.rating, r.timestamp, c.rating_counts FROM ratings r left join counts c on r.userId = c.userId WHERE c.rating_counts > 9") 
    training_only = spark.sql("SELECT r.userId, r.movieId, r.rating, r.timestamp, c.rating_counts FROM ratings r left join counts c on r.userId = c.userId WHERE c.rating_counts < 10") 

    # save to csv files
    to_split.coalesce(1).write.csv("full_tostratify.csv")
    training_only.coalesce(1).write.csv("full_lessthan10.csv")



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)

