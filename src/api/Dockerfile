FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /app/tools

# Copy requirements file
COPY src/api/requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY src/api/*.py /app/

# Copy tools directory
COPY src/tools /app/tools/

# Create data directory and prepare for hash database
RUN mkdir -p /app/data

# Copy the hash database file from the local src/data directory
# First, copy to a temporary location to avoid volume mount conflicts
COPY src/data/hashes_dphash_16.json /tmp/hash_database.json

# Verify the file was copied correctly and has content
RUN ls -la /tmp/hash_database.json && \
    echo "Hash database file size: $(stat -c %s /tmp/hash_database.json) bytes" && \
    if [ "$(stat -c %s /tmp/hash_database.json)" -gt "1000" ]; then \
        echo "Hash database file has content"; \
    else \
        echo "WARNING: Hash database file is too small, may be empty"; \
    fi && \
    # Create a copy in the data directory
    mkdir -p /app/data && \
    cp /tmp/hash_database.json /app/data/hashes_dphash_16.json && \
    chmod 644 /app/data/hashes_dphash_16.json && \
    echo "Hash database copied to /app/data/hashes_dphash_16.json"

# Set working directory
WORKDIR /app

# Make port 5000 available
EXPOSE 5000

# Run the application
CMD ["python", "simple_card_api.py"]
