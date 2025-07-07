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

# Create data directory and copy hash database
RUN mkdir -p /app/data
COPY src/data/hashes_dphash_16.json /app/data/

# Set working directory
WORKDIR /app

# Make port 5000 available
EXPOSE 5000

# Run the application
CMD ["python", "simple_card_api.py"]
