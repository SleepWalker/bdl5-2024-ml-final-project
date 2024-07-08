ARG REGISTRY=quay.io
ARG OWNER=jupyter
ARG BASE_CONTAINER=$REGISTRY/$OWNER/pytorch-notebook:2024-04-29
FROM $BASE_CONTAINER

RUN pip install --no-cache-dir \
  optuna \
  plotly \
  lightgbm \
  BorutaShap \
  "shap==0.44.1" \
  torchmetrics \
  torchinfo \
  dask \
  # postgress client
  psycopg2-binary
