FROM python:3.9

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /app/tools /app/data

# Copy requirements first for better caching
COPY src/api/requirements.txt /app/

# Install PyTorch first (separate step for better caching)
RUN pip install --no-cache-dir torch==1.9.0 torchvision==0.10.0

# Install MMCV and MMDetection with specific versions
# Use MMCV 1.3.17 which is compatible with MMDetection 2.20.0
RUN pip install --no-cache-dir mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

# Install MMDetection 2.20.0 and its dependencies
RUN pip install --no-cache-dir terminaltables
RUN pip install --no-cache-dir mmdet==2.20.0

# Install model configs
RUN pip install --no-cache-dir mmcls==0.23.0

# Install remaining requirements
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
