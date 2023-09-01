.PHONY: start wsgi_start

start:
	poetry run uvicorn main:app --reload

wsgi_start:
	poetry run gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker

