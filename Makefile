install:
	@pip install -r requirements.txt


run:
	@FLASK_APP=flask_app/app.py FLASK_ENV=development flask run --host=0.0.0.0 --port=3000


clean:
	find . -name "__pycache__" -exec rm -r {} +  
	find . -name "*.pyc" -delete
