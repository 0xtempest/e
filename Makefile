.PHONY: start wsgi_start

start:
	poetry run uvicorn main:app --reload

wsgi_start:
	poetry run gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker

make_env:
	python3 -m venv vllm_inference_env

make install:
	pip3 install -r requirements.txt
