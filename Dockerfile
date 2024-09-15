# Use Python 3.7 for your application
FROM python:3.7

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies necessary for h5py, TensorFlow, and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-cpp-103 \
    zlib1g-dev \
    gcc \
    g++ \
    python3-dev \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies using pip
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create the necessary directories and download the required datasets
RUN mkdir -p /usr/local/lib/python3.7/site-packages/aif360/data/raw/compas \
    && wget -O /usr/local/lib/python3.7/site-packages/aif360/data/raw/compas/compas-scores-two-years.csv \
    https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv

RUN mkdir -p /usr/local/lib/python3.7/site-packages/aif360/data/raw/german \
    && wget -O /usr/local/lib/python3.7/site-packages/aif360/data/raw/german/german.data \
    https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data \
    && wget -O /usr/local/lib/python3.7/site-packages/aif360/data/raw/german/german.doc \
    https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc

# Copy the rest of the application code into the container
COPY . .

# Expose the port your application will run on
EXPOSE 5000

# Command to run the Flask app (replace "app.py" with your entry point)
CMD ["python", "app.py"]
