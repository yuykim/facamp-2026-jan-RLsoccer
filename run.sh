#!/bin/bash

 docker run --rm -it \
    --platform linux/amd64 \
    --name gfr \
    -v ./workspace:/workspace \
    -w /workspace \
    -p 8888:8888 \
    hisplan/gfootball:2.10.2-facamp.1
