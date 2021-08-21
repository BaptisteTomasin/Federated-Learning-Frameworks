FROM python:3.8.5-slim

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm ./requirements.txt && \
    apt update && \
    apt install -y jq