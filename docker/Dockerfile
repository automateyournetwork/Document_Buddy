# Stage 1: Create a virtual environment, install requirements, copy core
# app files to the working directory
FROM python:3.11.8-slim AS builder
WORKDIR /app
ENV VIRTUAL_ENV=/app/.venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

COPY /requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY /document_buddy .

# Stage 2: Add the app user, copy the venv and app from the builder image,
# and launch the app.
FROM python:3.11.8-slim AS app
ARG APP_USERNAME
ARG APP_UID
ARG APP_GID=${APP_UID}

WORKDIR /app

RUN groupadd --gid ${APP_GID} ${APP_USERNAME} && \
    useradd --uid ${APP_UID} --gid ${APP_GID} -m ${APP_USERNAME} && \
    chown ${APP_USERNAME}:${APP_USERNAME} /app

COPY --from=builder --chown=${APP_USERNAME}:${APP_USERNAME} /app ./
USER ${APP_USERNAME}
ENV VIRTUAL_ENV=/app/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

CMD ["streamlit", "run", "document_buddy.py"]
