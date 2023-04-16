FROM python:3.9-slim as base

ARG PROJECT=src

RUN groupadd --gid 1000 user && adduser --disabled-password --gecos '' --uid 1000 --gid 1000 user
USER user

WORKDIR /home/user

RUN pip install --user poetry==1.4.2
ENV \
    PATH="/home/user/.local/bin:/home/user/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_HOME=/opt/poetry

COPY --chown=user:user ./pyproject.toml ./poetry.lock ./
RUN poetry install

FROM base as dev

COPY --chown=user:user ./$PROJECT /home/user/$PROJECT

CMD ["poetry", "run", "uvicorn", "--host", "0.0.0.0", "--port", "5000", "--reload", "src.app.main:app"]