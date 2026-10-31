# ThematicML — local web application image (no GPU, no remote AI services)
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY backend/ backend/
RUN pip install --no-cache-dir ".[sentiment,temporal,leiden,reports,api]"
COPY --from=ui /ui/dist frontend/dist
COPY sample_data/ sample_data/
ENV THEMATICML_HOME=/data THEMATICML_FRONTEND=/app/frontend/dist
VOLUME /data
EXPOSE 8000
# 0.0.0.0 is needed inside the container; publish it on the host loopback only (see README): the app has no authentication
CMD ["uvicorn", "thematicml_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
