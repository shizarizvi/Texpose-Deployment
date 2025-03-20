
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN pip install -U nltk

# Then copy the rest of the application files
COPY . /app

# Expose the port FastAPI runs on
EXPOSE 8000

# Define the entry point command
CMD ["uvicorn", "main:app", "--reload"]

