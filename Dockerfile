FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y curl unzip && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download and extract WordNet manually
RUN mkdir -p /usr/share/nltk_data/corpora && \
    curl -L -o /tmp/wordnet.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip && \
    unzip /tmp/wordnet.zip -d /usr/share/nltk_data/corpora && \
    curl -L -o /tmp/omw-1.4.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip && \
    unzip /tmp/omw-1.4.zip -d /usr/share/nltk_data/corpora && \
    rm -rf /tmp/*.zip

# Set environment variable for NLTK
ENV NLTK_DATA=/usr/share/nltk_data

EXPOSE 5000

CMD ["python3", "app.py"]
