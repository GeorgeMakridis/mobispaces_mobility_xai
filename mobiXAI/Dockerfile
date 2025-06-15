# Use an official Python runtime as a parent image
FROM python:3.9-slim

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y sudo dbus gcc python3-dev

RUN apt-get install --no-install-recommends -y \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install packages from requirements.txt first to leverage Docker cache
COPY requirements.txt /processor/requirements.txt
WORKDIR /processor
RUN pip3 install -r requirements.txt

COPY . /processor

RUN mkdir -p /mobi_data

EXPOSE 8881

# The command to run your application
# ENTRYPOINT ["python3", "/processor/app.py", "$@", "--"]

