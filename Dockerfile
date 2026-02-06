FROM python:3.11-slim


# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1



# Install system dependencies required for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*



# Set working directory
WORKDIR /app




# Copy requirements first for better layer caching
COPY requirements.txt .




# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt




# Copy the entire project
COPY . .




# Create directory for data collection if it doesn't exist
RUN mkdir -p Sign_data

# Expose port if needed for future web interface (optional)
# EXPOSE 5000




# Set environment variable for display (needed for GUI apps)
ENV DISPLAY=:0







# Default command - can be overridden
CMD ["python", "test.py"]
