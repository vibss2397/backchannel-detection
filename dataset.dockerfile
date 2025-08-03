# Use a specific Python version for reproducibility
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for building wheels like 'blis'
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements file and install dependencies
COPY requirements.dataset.txt .
RUN pip install --no-cache-dir -r requirements.dataset.txt

# Copy the script and data files into the container
COPY dataset/ ./dataset/

# This command will be executed when the container starts
CMD ["python", "dataset/dataset_gen.py"]