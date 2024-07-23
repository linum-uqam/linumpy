FROM python:3.11.4

WORKDIR /

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY linumpy-private /linumpy-private

RUN pip install -e /linumpy-private
