# Taller Sentimientos — NLP Laboratorio II

Sistema auditable de análisis de sentimientos sobre Sentiment140 con experimentación registrada en MLflow, modelo final en Model Registry y API FastAPI.

## URLs de entrega

- API pública: http://ec2-100-26-91-142.compute-1.amazonaws.com:8000
- Swagger: http://ec2-100-26-91-142.compute-1.amazonaws.com:8000/docs
- MLflow Tracking Server: http://ec2-100-26-91-142.compute-1.amazonaws.com:5000
- GitHub: https://github.com/MrDavid0607/TallerSentimientos

## Dataset

Dataset:

adilbekovich/Sentiment140Twitter

Revision:

b6037e127257d95b9b23d31f78b264b9ebe697fd

Tamaños:

- train: 1,360,000
- test: 240,000
- muestra experimental: 200,000

Protocolo:

- random seed: 42
- StratifiedKFold
- 3 folds
- shuffle=true

Artefactos MLflow:

- protocol/partitions.csv
- protocol/members.csv

## Pipeline seleccionado

### Preprocesamiento

P_ELONGATION

Incluye:

- lowercase
- URLs → url
- mentions → user
- normalización de espacios
- reducción de repeticiones de 3+ caracteres a 2

### Representación

TF-IDF:

ngram_range=(1,2)

### Clasificador

Logistic Regression.

## Modelo final

Entrenamiento:

1,360,000 registros.

Evaluación final:

240,000 registros de test.

Test Macro-F1:

0.8260640323943584

El conjunto test se evaluó una sola vez después de cerrar las decisiones experimentales.

## MLflow Model Registry

Modelo:

sentiment140

Alias:

champion

URI utilizada por la API:

models:/sentiment140@champion

La API no utiliza un modelo serializado local como fuente de inferencia.

## API

Endpoints:

- POST /api/v1/predict
- GET /audit/protocol
- GET /audit/runs
- GET /audit/contributions
- GET /audit/model
- GET /health

Ejemplo:

```bash
curl -X POST \
  http://ec2-100-26-91-142.compute-1.amazonaws.com:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":["i loved it","worst day ever"]}'
