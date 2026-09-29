FROM python:3.11-bookworm

WORKDIR /app/

# Install Poetry (version matches the one that generated poetry.lock)
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry POETRY_VERSION=2.5.1 python - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy poetry.lock* in case it doesn't exist in the repo
COPY ./pyproject.toml ./poetry.lock* /app/

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --without dev ; fi"

COPY . .

CMD ["sh", "./run.sh"]
