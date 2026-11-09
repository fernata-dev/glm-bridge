FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY glm_bridge ./glm_bridge

RUN useradd --system --uid 10001 bridge
USER bridge

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4)" || exit 1

CMD ["python", "-m", "glm_bridge", "serve"]
