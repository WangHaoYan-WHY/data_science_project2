import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# calculate the mean without the 0
def mean(row):
    row = row[row > 0]
    return sum(row) / len(row)


neighboors_num = 30
user_id = 1
movie_id = 2

# read data from csv
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
# get user-movie matrix
user_movie_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
# each user's mean rating
_user_mean_rate = user_movie_matrix.apply(mean, axis=1)
_movie_mean_rate = user_movie_matrix.T.apply(mean, axis=1)
# calculate the user similarity and movies similarity
user_similarity = pd.DataFrame(cosine_similarity(user_movie_matrix), index=user_movie_matrix.index, columns=user_movie_matrix.index)
movie_similarity = pd.DataFrame(cosine_similarity(user_movie_matrix.T), index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# get the most similarest instance between neignbors, the neighbors' number here is 30
_user_similarest = pd.DataFrame(index=user_similarity .index, columns=range(1, neighboors_num))
for index in _user_similarest.index:
    _user_similarest.loc[index, :neighboors_num - 1] = user_similarity .loc[:, index].sort_values(ascending=False)[
                                                            1:neighboors_num].index

_movie_similarest = pd.DataFrame(index= movie_similarity .index, columns=range(1, neighboors_num))
for index in _movie_similarest.index:
    _movie_similarest.loc[index, :neighboors_num - 1] = movie_similarity .loc[:, index].sort_values(ascending=False)[
                                                             1:neighboors_num].index

# predict the user-based CF
user_prediction = 0
sums = user_similarity .loc[user_id, _user_similarest.loc[user_id, :]].sum()
for user in _user_similarest.loc[user_id, :]:
    if user_movie_matrix.loc[user, movie_id] != 0:
        user_prediction += user_similarity .loc[user_id, user] * (
                user_movie_matrix.loc[user, movie_id] - _user_mean_rate[user])
user_prediction /= sums
user_prediction += _user_mean_rate[user_id]
print('user-based rating is {}'.format(user_prediction))

# predict the item-based CF
movie_prediction = 0
sums = movie_similarity.loc[movie_id, _movie_similarest.loc[movie_id, :]].sum()
for movie in _movie_similarest.loc[movie_id, :]:
    if user_movie_matrix.loc[user_id, movie] != 0:  # For here, 0 is that user doesn't give the rate
        movie_prediction += movie_similarity.loc[movie_id, movie] * (user_movie_matrix.loc[user_id, movie] - _movie_mean_rate[movie])
movie_prediction /= sums
movie_prediction += _movie_mean_rate[movie_id]
print('item-based rating is {}'.format(movie_prediction))

