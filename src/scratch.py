import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

ratings_contents = pd.read_table("data/u.data",
                                 names=["user", "movie", "rating",
                                        "timestamp"])
highest_user_id = ratings_contents.user.max()
highest_movie_id = ratings_contents.movie.max()
ratings_as_mat = sparse.lil_matrix((highest_user_id, highest_movie_id))

for _, row in ratings_contents.iterrows():
    # subtract 1 from id's due to match 0 indexing
    ratings_as_mat[row.user - 1, row.movie - 1] = row.rating
ratings_contents, ratings_as_mat
# so, the ratings_as_mat has the user's rating per movie

def make_cos_sim_and_neighborhoods(ratings_mat, neighborhood_size):
    '''
    Accepts a 2 dimensional matrix ratings_mat, and an integer neighborhood_size.
    Returns a tuple containing:
        - items_cos_sim, an item-item matrix where each element is the
        cosine_similarity of the items at the corresponding row and column. This
        is a square matrix where the length of each dimension equals the number
        of columns in ratings_mat.
        - neighborhood, a 2-dimensional matrix where each row is the neighborhood
        for that item. The elements are the indices of the n (neighborhood_size)
        most similar items. Most similar items are at the end of the row.
    '''
    # hokay so you are taking the transpose of the ratings matrix.
    # and assigning it to items_cos_sim
    items_cos_sim = cosine_similarity(ratings_mat.T)

    least_to_most_sim_indexes = np.argsort(items_cos_sim, 1)
    neighborhood = least_to_most_sim_indexes[:, -neighborhood_size:]
    return items_cos_sim, neighborhood

print make_cos_sim_and_neighborhoods(ratings_as_mat,neighborhood_size=75)
