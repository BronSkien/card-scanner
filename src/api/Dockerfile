FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /app/tools /app/data

# Copy requirements first for better caching
COPY src/api/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY src/api/*.py /app/

# Copy tools directory
COPY src/tools /app/tools/

# Create data directory and copy hash database
RUN mkdir -p /app/data
COPY src/data/hashes_dphash_16.json /app/data/

# Make port 5000 available
EXPOSE 5000

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "simple_card_api:app"]
