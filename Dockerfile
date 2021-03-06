# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

RUN apt-get update
RUN apt-get install libgtk2.0-dev -y

# Install production dependencies.
COPY . .
RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["python", "model.py"]
