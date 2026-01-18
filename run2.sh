#!/bin/bash

docker run --rm -it \
   --platform linux/amd64 \
   -v ./workspace:/workspace \
   -w /workspace \
   gfootball:2.10.2-facamp.1
