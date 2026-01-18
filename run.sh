#!/bin/bash

 docker run --rm -it \
    --platform linux/amd64 \
    --name gfr \
    -v ./workspace:/workspace \
    -w /workspace \
    -p 8888:8888 \
    -p 6006:6006 \
    gfootball:2.10.2-facamp.1
