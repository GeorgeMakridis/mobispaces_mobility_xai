# Use a slim version of the OpenJDK image as the base
FROM openjdk:slim

# Install Python3 and pip for Debian-based images
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install packages from requirements.txt first to leverage Docker cache
COPY requirements.txt /processor/requirements.txt
WORKDIR /processor
RUN pip3 install -r requirements.txt

COPY . /processor

RUN mkdir -p /mobi_data

EXPOSE 8880

# The command to run your application
# ENTRYPOINT ["python3", "/processor/app.py", "$@", "--"]
