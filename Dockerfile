FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only the necessary files for installing dependencies first
COPY requirements.txt /tmp/

# Install dependencies and clean up
RUN apt update && \
    apt install -y --no-install-recommends htop libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . /app

# Specify the command to run the application
CMD ["uvicorn", "main:app", "--port=8000", "--host=0.0.0.0"]