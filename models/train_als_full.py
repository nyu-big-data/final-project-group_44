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


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

     # Read in training data
    train = spark.read.csv(f'hdfs:/user/{netID}/pub/ratings_all_train.csv', header='true', schema='index INT, userId INT,movieId INT,rating DOUBLE,timestamp INT')


    als = ALS(rank = 30, maxIter=20, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating",\
                        nonnegative = True, implicitPrefs = True, coldStartStrategy="drop", seed=42)
    start = time.time()
    model = als.fit(train)
    stop = time.time()
    model.write().overwrite().save(f"hdfs:/user/{netID}/ALS_model_full_rank30_reg0.1")

    # print time
    print('total time for als on full set:', stop - start)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
