FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure nltk stopwords are downloaded
RUN python -c "import nltk; nltk.download('stopwords')"

# Expose the port Flask runs on
EXPOSE 5000

# Define the entry point command
CMD ["python", "main.py"]
