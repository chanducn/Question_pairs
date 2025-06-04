# Use an official Python 3.10 image from Docker Hub
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . /app

# Install the dependencies
RUN pip install -r requirements.txt
RUN mkdir -p /usr/share/nltk_data
RUN python3 -m nltk.downloader -d /usr/share/nltk_data wordnet
ENV NLTK_DATA=/usr/share/nltk_data



# Expose the port FastAPI will run on
EXPOSE 5000

# Command to run the FastAPI app
CMD ["python3", "app.py"]
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]