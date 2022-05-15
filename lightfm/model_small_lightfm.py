import pandas as pd
from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn

train = pd.read_csv('/Users/jennaelizabethellis/Desktop/NYU/bigdata/finalproject/final-project-group_44/data/ratings_small_train.csv',)
train = train.drop(columns='Unnamed: 0')
test = pd.read_csv('/Users/jennaelizabethellis/Desktop/NYU/bigdata/finalproject/final-project-group_44/data/ratings_small_test.csv',)
test = test.drop(columns='Unnamed: 0')

train = train.rename(columns={"userId": "user", "movieId": "item"})
test = test.rename(columns={"userId": "user", "movieId": "item"})

algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(50)

def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs

all_recs = []
all_recs.append(eval('ItemItem', algo_ii, train, test))
all_recs.append(eval('ALS', algo_als, train, test))

test_data = []
test_data.append(test)

all_recs = pd.concat(all_recs, ignore_index=True)
all_recs.head()

test_data = pd.concat(test_data, ignore_index=True)
test_data.head()

rla = topn.RecListAnalysis()
rla.add_metric(topn.ndcg)
results = rla.compute(all_recs, test_data)

results.groupby('Algorithm').ndcg.mean()

results.groupby('Algorithm').ndcg.mean().plot.bar()
