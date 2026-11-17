FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY polyengine ./polyengine
RUN pip install --no-cache-dir .

# SQLite DB + logs live here; mount a volume so state survives restarts.
VOLUME ["/app/data"]
ENV PYTHONUNBUFFERED=1

# Paper mode by default. For live: set PE_LIVE_TRADING=true and override CMD with `run --live`.
ENTRYPOINT ["polyengine"]
CMD ["run"]
