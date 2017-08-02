import graphlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    r_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    pandas_data = pd.read_csv('data/u.data', sep='\t', names =r_cols,encoding='latin-1')
    return graphlab.SFrame(pandas_data[['user_id', 'item_id', 'rating']])

def one_predict():
    '''predicting for user 1, movie 100.'''
    movie_array = np.array(m1.coefficients['item_id'][100]['factors'])
    user_array = m1.coefficients['user_id'][1]['factors']
    return np.dot(movie_array, user_array) + m1.coefficients['intercept']

if __name__ == '__main__':
    sf = load_data()
    m1 = graphlab.recommender.factorization_recommender.create(sf, target ='rating',solver='als')
    one_datapoint_sf = graphlab.SFrame({'user_id': [1], 'item_id': [100]})
    print m1.predict(one_datapoint_sf)
    #our result was 3.5299
    m1.list_fields()
    coef = m1.get('coefficients')
    print one_predict()
    #predictions = m1.coefficients['item_id'][1]['factors'][100]
