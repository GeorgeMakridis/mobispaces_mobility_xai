# Use an official Python runtime as a parent image
FROM python:3.8-slim

RUN apt-get update && apt-get install -y sudo dbus gcc python3-dev

WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit --upgrade

COPY . /app
RUN mkdir -p /mobi_data

# Make Streamlit's port available
EXPOSE 8501
ENV PORT 8501

# Define environment variable
# Ensure Streamlit uses the network interface to serve the app, making it accessible outside the container
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

### Run streamlit when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
# ENTRYPOINT ["python3", "flask_app.py", "$@", "--"]
