# Runs the trading agent and its dashboard in a container (paper trading unless you configure otherwise).
#
# The agent's state (trade journal, trained model, downloaded prices) lives in /data. Mount a
# persistent volume there, or that state is lost whenever the container is replaced.
#
# Required environment: DASHBOARD_TOKEN (password for the dashboard, at least 16 characters).
# Optional: PORT (default 8000), BTCAGENT_SET (config overrides, see README), ANTHROPIC_API_KEY.
ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BTCAGENT_STATE_DIR=/data

RUN useradd --system --create-home --home-dir /home/btcagent --shell /usr/sbin/nologin btcagent \
    && mkdir -p /data \
    && chown btcagent:btcagent /data

WORKDIR /app

# Install the exact dependency versions this project was tested with, then the app itself.
COPY deploy/constraints.txt /tmp/constraints.txt
COPY pyproject.toml README.md ./
COPY btcagent ./btcagent
RUN pip install -c /tmp/constraints.txt . \
    && rm -rf /app/btcagent /app/build /app/*.egg-info /app/pyproject.toml /app/README.md /tmp/constraints.txt

COPY config.toml ./
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '8000'), timeout=4)"

ENTRYPOINT ["/entrypoint.sh"]
