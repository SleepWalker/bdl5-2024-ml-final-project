#! /bin/bash

docker build --tag 'jupyter-lab' .

docker run --rm \
  -p 8888:8888 \
  -v "${PWD}":/home/jovyan/work \
  --network="host" \
  jupyter-lab