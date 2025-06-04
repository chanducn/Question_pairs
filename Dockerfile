FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download both required NLTK packages
RUN mkdir -p /usr/share/nltk_data && \
    python3 -m nltk.downloader -d /usr/share/nltk_data wordnet omw-1.4 && \
    ls /usr/share/nltk_data/corpora/wordnet

# Set environment variable
ENV NLTK_DATA=/usr/share/nltk_data

EXPOSE 5000

CMD ["python3", "app.py"]
