from item_item_prototype import get_ratings_data

class ItemItemRecommender(object):
    def __init__(self, neighborhood_size):
        '''
        Initialize the parameters of the model.
        '''
        self.neighborhood_size = neighborhood_size
        self.ratings_mat = None
        self.neighborhood = None
        self.items_cos_sim = None

    def fit(self, X):
        '''
        Implement the model and fit it to the data passed as an argument.

        Store objects for describing model fit as class attributes.
        '''
        self.ratings_mat = self._set_neighborhoods(X)

    def _set_neighborhoods(self, ratings_mat):
        '''
        Get the items most similar to each other item.

        Should set a class attribute with a matrix that is has
        number of rows equal to number of items and number of
        columns equal to neighborhood size. Entries of this matrix
        will be indexes of other items.

        You will call this in your fit method.
        '''
        self.items_cos_sim = cosine_similarity(ratings_mat.T)
        least_to_most_sim_indexes = np.argsort(items_cos_sim, 1)
        self.neighborhood = least_to_most_sim_indexes[:, self.neighborhood_size:]

    def pred_one_user(self):
        '''
        Accept user id as arg. Return the predictions for a single user.

        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''
        pass

    def pred_all_users(self):
        '''
        Repeated calls of pred_one_user, are combined into a single matrix.
        Return value is matrix of users (rows) items (columns) and predicted
        ratings (values).

        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''
        pass

    def top_n_recs(self):
        '''
        Take user_id argument and number argument.

        Return that number of items with the highest predicted ratings, after
        removing items that user has already rated.
        '''
        pass

if __name__ == '__main__':
    iir = ItemItemRecommender(neighborhood_size=20)
    iir.fit()
