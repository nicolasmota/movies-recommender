import json
import tempfile
import numpy as np
import pandas as pd

from native_recommender.item_similiarity.main import NativeRecommender


class TestNativeRecommender:

    def get_temp_file(self, value):
        temp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        temp_file.write(value)
        temp_file.close()
        return temp_file

    def test_get_file(self):
        temp_file = self.get_temp_file(b'{"test": {"test1": 100}}')
        nrecommender = NativeRecommender(temp_file.name)
        assert nrecommender

    def test_get_movies(self):
        expected_json = b'{"movies": [{"1": "Test"}, {"2": "Test2"}]}'
        temp_file = self.get_temp_file(expected_json)
        nrecommender = NativeRecommender(temp_file.name)

        assert (
            nrecommender._get_movies() ==
            json.loads(expected_json)['movies']
        )

    def test_normalize_dataset_should_return_a_dataframe(self):
        users_json = b'{"users": [{"user_id": 1, "movies": [1, 2, 3, 4]}]}'
        temp_file = self.get_temp_file(users_json)
        nrecommender = NativeRecommender(temp_file.name)

        assert isinstance(nrecommender._normalize_dataset(), pd.DataFrame)
        assert nrecommender._normalize_dataset().shape == (4, 3)

    def test_should_create_a_sparse_matrix(self):
        users_json = b'''
            {"users": [
                {"user_id": 1, "movies": [1, 2, 3, 4]},
                {"user_id": 2, "movies": [3, 2, 4]}
            ]}
        '''
        temp_file = self.get_temp_file(users_json)
        nrecommender = NativeRecommender(temp_file.name)
        sparse_data = nrecommender._get_sparse_data()

        assert isinstance(sparse_data, np.ndarray)
        assert sparse_data.shape == (2, 4)

    def test_should_create_train_test_data(self):
        users_json = b'''
            {"users": [
                {"user_id": 1, "movies": [1, 2, 3, 4]},
                {"user_id": 2, "movies": [3, 2, 4]}
            ]}
        '''
        temp_file = self.get_temp_file(users_json)
        nrecommender = NativeRecommender(temp_file.name)
        sparse_data = nrecommender._get_sparse_data()
        train, test = nrecommender._create_train_test_data(sparse_data)

        assert isinstance(train, np.ndarray)
        assert train.shape == (2, 4)

        assert isinstance(test, np.ndarray)
        assert test.shape == (2, 4)

    def test_should_get_correlation_item(self):
        users_json = b'''
            {"users": [
                {"user_id": 1, "movies": [1, 2, 3, 4]},
                {"user_id": 2, "movies": [3, 2, 4]}
            ]}
        '''
        temp_file = self.get_temp_file(users_json)
        nrecommender = NativeRecommender(temp_file.name)
        sparse_data = nrecommender._get_sparse_data()
        train, _ = nrecommender._create_train_test_data(sparse_data)

        item_correlation = nrecommender._get_item_correlation(train)

        assert item_correlation.shape == (4, 4)
        assert not (item_correlation == 0).all()

    def test_should_return_top_k_movies(self):
        users_json = b'''
            {
            "users": [
                {"user_id": 1, "movies": [1, 2, 3, 4]},
                {"user_id": 2, "movies": [3, 2, 4]}
            ],
            "movies": {
                "1": "Toy Story (1995)",
                "2": "GoldenEye (1995)",
                "3": "Twelve Monkeys (1995)",
                "4": "Richard III (1995)",
                "5": "Seven (Se7en) (1995)",
                "6": "Usual Suspects, The (1995)"
             }
        }
        '''
        temp_file = self.get_temp_file(users_json)
        nrecommender = NativeRecommender(temp_file.name)
        sparse_data = nrecommender._get_sparse_data()
        train, _ = nrecommender._create_train_test_data(sparse_data)

        item_correlation = nrecommender._get_item_correlation(train)
        movie_id = 2

        for i in range(1, 5):
            movies = nrecommender._get_top_k_movies(
                item_correlation, movie_id, i
            )
            assert len(movies) == i

    def test_should_recommend_k_movies(self):
        users_json = b'''
            {
            "users": [
                {"user_id": 1, "movies": [1, 2, 3, 4]},
                {"user_id": 2, "movies": [3, 2, 4]}
            ],
            "movies": {
                "1": "Toy Story (1995)",
                "2": "GoldenEye (1995)",
                "3": "Twelve Monkeys (1995)",
                "4": "Richard III (1995)",
                "5": "Seven (Se7en) (1995)",
                "6": "Usual Suspects, The (1995)"
             }
        }
        '''
        temp_file = self.get_temp_file(users_json)
        nrecommender = NativeRecommender(temp_file.name)
        movie_id = 2
        movies = nrecommender.recommend(movie_id, 4)
        """ Based on id 2 it will pick up the id that has correlation
        in other users, until the k """
        assert sorted(movies) == sorted([
            'GoldenEye (1995)',
            'Richard III (1995)',
            'Twelve Monkeys (1995)',
            'Toy Story (1995)'
        ])
