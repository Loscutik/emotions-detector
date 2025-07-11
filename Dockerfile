# syntax docker/dockerfile:1

# Alpine is chosen for its small footprint
# compared to Ubuntu
FROM continuumio/miniconda3:latest
LABEL progect-name="emotion-detector"
LABEL version="1.0.0"
LABEL authors="Olena Budarahina"

ARG PORT=8888
ENV PORT=$PORT
WORKDIR /app
COPY . .

RUN conda install --channel conda-forge --yes --file ./environments/environment.txt
RUN conda clean --all --yes


EXPOSE $PORT
CMD jupyter lab --port $PORT --no-browser --ip='*'  --allow-root