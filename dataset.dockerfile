# Use a specific Python version for reproducibility
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for building wheels like 'blis'
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script and data files into the container
COPY dataset_gen.py .
COPY synthetic_dataset_claude.csv .
COPY synthetic_dataset_gemini.csv .

# This command will be executed when the container starts
CMD ["python", "dataset_gen.py"]