# Use an official Python 3.10 image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir -p /usr/share/nltk_data
RUN python3 -m nltk.downloader -d /usr/share/nltk_data wordnet


ENV NLTK_DATA=/usr/share/nltk_data

# Expose app port
EXPOSE 5000

# Start the app
CMD ["python3", "app.py"]
