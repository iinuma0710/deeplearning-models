FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update && \
    apt install -y nodejs npm && \
    npm install n -g && \
    n stable && apt purge -y nodejs npm