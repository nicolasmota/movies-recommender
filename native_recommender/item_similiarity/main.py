import argparse
import json
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances


class NativeRecommender:

    def __init__(self, data_path):
        self.data_file = self._get_data_file(data_path)

    def _get_data_file(self, data_path):
        """ Return a json object containing the file content """

        return json.load(open(data_path))

    def _get_movies(self):
        """ Return a dict of movies """

        return self.data_file['movies']

    def _normalize_dataset(self):
        """ Create DataFrame including rating 5 for all watched movies and
        split to row each movie_id by user """

        new_data = []
        columns = ['user_id', 'movie_id', 'rating']
        for line in self.data_file['users']:
            movies_by_user = [
                {'user_id': line['user_id'], 'movie_id': movie_id, 'rating': 5}
                for movie_id in line['movies']
            ]
            new_data.extend(movies_by_user)
        return pd.DataFrame(new_data, columns=columns)

    def _get_sparse_data(self):
        """ Read a normalized dataset and create a sparse matrix
        containing the ratings """
        data = self._normalize_dataset()

        n_users = data.user_id.unique().shape[0]
        n_movies = data.movie_id.unique().shape[0]

        ratings = np.zeros((n_users, n_movies))

        for row in data.itertuples():
            ratings[row[1]-1, row[2]-1] = row[3]
        return ratings

    def _create_train_test_data(self, ratings):
        test = np.zeros(ratings.shape)
        train = ratings.copy()
        for user in range(ratings.shape[0]):
            test_ratings = np.random.choice(
                ratings[user, :].nonzero()[0],
                size=5
            )
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]

        assert(np.all((train * test) == 0))
        return train, test

    def _get_item_correlation(self, train):
        item_correlation = 1 - pairwise_distances(
            train.T,
            metric='correlation'
        )
        item_correlation[np.isnan(item_correlation)] = 0.
        return item_correlation

    def _get_top_k_movies(self, similarity, movie_id, k):
        """ Return a list of k recommended movies """
        return [
            self._get_movies()[str(x+1)]
            for x in np.argsort(similarity[movie_id-1,:])[:-k-1:-1]
        ]

    def recommend(self, movie_id, k=5):
        ratings = self._get_sparse_data()
        train, _ = self._create_train_test_data(ratings)
        item_correlation = self._get_item_correlation(train)
        return self._get_top_k_movies(item_correlation, movie_id, k)


if __name__ == "__main__":
    k = 5
    parser = argparse.ArgumentParser(
        description="Native Recommender movies Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input', type=str, required=True,
        dest='inputfile', help='data file path')

    parser.add_argument(
        '--movie_id', type=int, required=True,
        dest='movie_id', help='Movie ID')
    parser.add_argument(
        '--k', type=int, required=False,
        dest='k', help='Number of recommended movies'
    )

    args = parser.parse_args()
    if args.k:
        k = args.k
    native_recommender = NativeRecommender(args.inputfile)
    movie_ids = native_recommender.recommend(args.movie_id, k)
    print(movie_ids)
