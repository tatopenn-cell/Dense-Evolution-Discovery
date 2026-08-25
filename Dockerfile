# Development / CI environment for Dense-Evolution-Discovery -- reproduces
# the environment the cross-validation suite is run in (see
# requirements-lock.txt). Not an image meant to be published or run as a
# service: this repo is a research log, not a library -- the container
# exists so an experiment's result can be reproduced on another machine
# exactly, without chasing "which numpy/jax version did I have" by hand.
FROM python:3.12-slim

WORKDIR /workspace

COPY requirements-lock.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-lock.txt

COPY . .

CMD ["pytest", "tests/", "-v", "--tb=short"]
