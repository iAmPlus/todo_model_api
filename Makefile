.PHONY: venv install clean dep rebuild

email_CONFIG_FILE ?= config/setup.yaml
email_BIND_PORT := $(shell config/read-option -s ${email_CONFIG_FILE} email_BIND_PORT 64554)
email_WORKER_CONCURRENCY := $(shell config/read-option -s ${email_CONFIG_FILE} classifier_WORKER_CONCURRENCY 1)

venv:
	python3 -m venv venv

install:
	python3 setup.py install

clean:
	rm -rf build dist *.egg-info

dep:
	pip install -r requirements.txt

rebuild: clean dep install

start-email-classify-tflite-docker:
	gunicorn -t 9000 -w${email_WORKER_CONCURRENCY} -b 0.0.0.0:${email_BIND_PORT} tflite_classifier.web_test:app