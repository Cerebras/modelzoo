#!/bin/bash

set -xe

docker build -t nemo docker
docker save nemo -o /tmp/nemo.img
singularity build ./nemo.sif docker-archive:///tmp/nemo.img
