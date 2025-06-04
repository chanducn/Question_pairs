# Use Python base image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set up NLTK data
RUN mkdir -p /usr/share/nltk_data && \
    python3 -m nltk.downloader -d /usr/share/nltk_data wordnet omw-1.4

# Tell Python where to find NLTK data
ENV NLTK_DATA=/usr/share/nltk_data

# Expose port
EXPOSE 5000

# Run your app
CMD ["python3", "app.py"]
