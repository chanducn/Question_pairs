# Use an official Python 3.10 image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data to standard path
RUN python3 -m nltk.downloader wordnet omw-1.4

# OPTIONAL: set environment variable just in case
ENV NLTK_DATA=/root/nltk_data

# Expose app port
EXPOSE 5000

# Start the app
CMD ["python3", "app.py"]
