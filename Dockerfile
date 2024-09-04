# Use an official Miniconda runtime as a parent image
FROM python:3.11.0-slim

# Python and app settings
ENV PORT=8000 \
    APP_HOME=/app \
    PYTHONPATH=${APP_HOME}

# Set the working directory in the container
WORKDIR ${APP_HOME}

# Install Tesseract and required dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy just the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the desired port
EXPOSE ${PORT}

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
