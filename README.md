# Movies Recommender


Movies Recommender is a command line python application that recommends movies based on movies that have been watched by other users

## Installation

1 - Install pyenv:

[pyenv](https://github.com/pyenv/pyenv#installation)

2 - Create a Virtualenv using python 3:

```
pyenv virtualenv 3.6.0 native_recommender
``` 

3 - Activate the virtualenv:

```
pyenv activate native_recommender
```

4 - install the dependencies:

```
make install
```

## Testing

To run the tests, run:
```
make test
```

## Running

To run the recommendation application, there are some parameters, they are in the following format (parameter, type, optional):

`
input: path, required
`

* You need to pass the path of the file that contains the dataset

`
movie_id: int, required
`

It is necessary to inform the id of the movie that will be based the recommendations

`
k: int, optional
`

K is the number of movies returned by the recommendations


To run the application:
```
make run input=data/movies.json movie_id=1 k=5
```
