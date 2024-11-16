#!/bin/sh

docker run -d -p 127.0.0.1:7070:9090 -v $(pwd)/prom.yaml:/etc/prometheus/prometheus.yml prom/prometheus

docker run -d -ti -v $(pwd)/data.yaml:/config.yml -p 127.0.0.1:9000:9000 littleangryclouds/prometheus-data-generator
