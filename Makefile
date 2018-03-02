clean:
	@find . -name "*.pyc" | xargs rm -rf
	@find . -name "*.pyo" | xargs rm -rf
	@find . -name "__pycache__" -type d | xargs rm -rf
	@rm -f .coverage
	@rm -rf htmlcov/
	@rm -f coverage.xml
	@rm -f *.log

test: clean
	@py.test -x -v

install:
	@pip install -r requirements.txt --upgrade

run:
	python movies_recommender/item_similiarity/main.py --input ${input} --movie_id ${movie_id} --k ${k}
