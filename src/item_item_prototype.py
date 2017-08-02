from time import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

class ItemItemRecommender:
    def __init__(self, neighborhood_size):
        self.neighborhood_size =neighborhood_size

<<<<<<< HEAD
def get_ratings_data():
    '''
    Returns a tuple containing:
        - a dataframe of ratings
        - a sparse matrix where rows correspond to users and columns correspond
        to movies. Each element is the user's rating for that movie.
    '''
    ratings_contents = pd.read_table("../data/u.data",
                                     names=["user", "movie", "rating",
                                            "timestamp"])
    highest_user_id = ratings_contents.user.max()
    highest_movie_id = ratings_contents.movie.max()
    ratings_as_mat = sparse.lil_matrix((highest_user_id, highest_movie_id))
    # going through the rows of the original df

    for _, row in ratings_contents.iterrows():
        # subtract 1 from id's due to match 0 indexing
        ratings_as_mat[row.user - 1, row.movie - 1] = row.rating
    return ratings_contents, ratings_as_mat

    ''' so it goes a lil somethin like this:
           |  movie  |  movie  |  movie
    users  |  rating |  rating |  rating
    users  |  rating |  rating |  rating
    '''

=======

    def fit(self,data):
        '''
        Returns a tuple containing:
            - a dataframe of ratings
            - a sparse matrix where rows correspond to users and columns correspond
            to movies. Each element is the user's rating for that movie.
        '''
        self.ratings_contents = data 
        highest_user_id = self.ratings_contents.user.max()
        highest_movie_id = self.ratings_contents.movie.max()
        self.ratings_mat = sparse.lil_matrix((highest_user_id, highest_movie_id))
        for _, row in self.ratings_contents.iterrows():
            # subtract 1 from id's due to match 0 indexing
            self.ratings_mat[row.user - 1, row.movie - 1] = row.rating
>>>>>>> pair


    def make_cos_sim_and_neighborhoods(self):
        '''
        Returns a tuple containing:
            - items_cos_sim, an item-item matrix where each element is the
            cosine_similarity of the items at the corresponding row and column. This
            is a square matrix where the length of each dimension equals the number
            of columns in ratings_mat.
            - neighborhood, a 2-dimensional matrix where each row is the neighborhood
            for that item. The elements are the indices of the n (neighborhood_size)
            most similar items. Most similar items are at the end of the row.
        '''
        self.items_cos_sim = cosine_similarity(self.ratings_mat.T)
        least_to_most_sim_indexes = np.argsort(self.items_cos_sim, 1)
        self.neighborhood = least_to_most_sim_indexes[:, -self.neighborhood_size:]
        return self.items_cos_sim, self.neighborhood

    def pred_one_user(self,user_id,printit=False,printruntime=False):
        '''
        Returns the predicted ratings for all items for a given user.
        '''
        start = time()
        n_items = self.ratings_mat.shape[1]
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        # Just initializing so we have somewhere to put rating preds
        output = np.zeros(n_items)
        for item_to_rate in range(n_items):
            relevant_items = np.intersect1d(self.neighborhood[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)
                                        # assume_unique speeds up intersection op
            output[item_to_rate] = self.ratings_mat[user_id, relevant_items] * \
                self.items_cos_sim[item_to_rate, relevant_items] / \
                self.items_cos_sim[item_to_rate, relevant_items].sum()
        output = np.nan_to_num(output)
        if printruntime is True:
            end = time()
            print end - start
        if printit is True:
            return output
    

    def pred_all_users(self):
        #output = np.zeros(self.ratings_mat.shape)
        #for row in xrange(output.shape[0]):
        #    output[row,:] = self.pred_one_user(row)
        #return output
        
        vpreder = np.vectorize(self.pred_one_user)
        return vpreder(self.ratings_contents.user)


    def top_rec_idx(self, user_id, n_arguments):
        full = np.argsort(pred_one_user(user_id,printit=True))
        u_input = self.ratings_mat[user_id].nonzero()[1] #array of indices the user already rated (TB ignored)
        return np.argsort(pred_one_user(user_id,printit=True))[-n_arguments:]


if __name__ == '__main__':
    data =  pd.read_table("data/u.data", names=["user", "movie", "rating", "timestamp"])
    model = ItemItemRecommender(75)
    model.fit(data)
    
    cos_sim, nbrhoods = model.make_cos_sim_and_neighborhoods()
    user_1_preds = model.pred_one_user(1)
    # Show predicted ratings for user #1
    print user_1_preds
    print model.pred_all_users()
