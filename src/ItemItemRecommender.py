import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from time import time


class ItemItemRecommender(object):
    def __init__(self, neighborhood_size):
        '''
        Initialize the parameters of the model.
        '''
        self.neighborhood_size = neighborhood_size
        self.ratings_mat = None
        self.neighborhood = None
        self.items_cos_sim = None

    def fit(self, ratings_mat):
        '''
        Implement the model and fit it to the data passed as an argument.

        Store objects for describing model fit as class attributes.
        '''
        self.ratings_mat = ratings_mat
        self.n_users = ratings_mat.shape[0]
        self.n_items = ratings_mat.shape[1]
        self.items_sim_mat = cosine_similarity(self.ratings_mat.T)

        self._set_neighborhoods()

    def _set_neighborhoods(self):
        '''
        Get the items most similar to each other item.

        Should set a class attribute with a matrix that is has
        number of rows equal to number of items and number of
        columns equal to neighborhood size. Entries of this matrix
        will be indexes of other items.

        You will call this in your fit method.
        '''
        least_to_most_sim_indexes = np.argsort(self.items_sim_mat, 1)
        self.neighborhoods = least_to_most_sim_indexes[:, self.neighborhood_size:]

    def pred_one_user(self, user_id):
        '''
        Accept user id as arg. Return the predictions for a single user.

        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        holder = np.zeros(self.n_items)
        for item in range(self.n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item], items_rated_by_this_user, assume_unique=True)
            holder[item] = self.ratings_mat[user_id, relevant_items].sum()
        cleaned_holder = np.nan_to_num(holder)
        return cleaned_holder

    def pred_all_users(self):
        '''
        Repeated calls of pred_one_user, are combined into a single matrix.
        Return value is matrix of users (rows) items (columns) and predicted
        ratings (values).

        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''
        all_ratings = [self.pred_one_user(user_id) for user_id in range(self.n_users)]
        return np.array(all_ratings)

    def top_n_recs(self, user_id, n):
        '''
        Take user_id argument and number argument.

        Return that number of items with the highest predicted ratings, after
        removing items that user has already rated.
        '''
        predicted_ratings = self.pred_one_user(user_id)
        item_index_by_predicted_ratings = list(np.argsort(predicted_ratings))
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        unrated = [item for item in item_index_by_predicted_ratings if item not in items_rated_by_this_user]
        return unrated[-n:]

def get_ratings_data():
    ratings = pd.read_table('/Users/CamillaNawaz/Google Drive/Galvanize/item-recommendation-system/data/u.data', names=["user", "movie", "rating", "timestamp"])
    highest_user_id = ratings.user.max()
    highest_movie_id = ratings.movie.max()
    ratings_matrix = sparse.lil_matrix((highest_user_id, highest_movie_id))
    for _, row in ratings.iterrows():
        ratings_matrix[row.user-1, row.movie-1] = row.rating
    return ratings, ratings_matrix


if __name__ == '__main__':
    ratings_data, ratings_matrix = get_ratings_data()
    my_recommender = ItemItemRecommender(neighborhood_size=20)
    my_recommender.fit(ratings_matrix)
    # for example, ....
    user12_predict = my_recommender.pred_one_user(user_id=12)
    print user12_predict[:100]
    print my_recommender.top_n_recs(12, 20)
