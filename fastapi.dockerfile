# Start with a clean Python environment
FROM python:3.9

# Create a directory inside the container
WORKDIR /code

# 1. COPY ONLY the FastAPI requirements
#    (We are NOT copying any training-specific libraries)
COPY ./fastapi/requirements.txt /code/requirements.txt

# 2. INSTALL ONLY those FastAPI requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 3. COPY ONLY the code needed for the app to run
COPY ./sharedlib /code/sharedlib
COPY ./fastapi/app /code/app
COPY ./fastapi/src /code/src

# The training/ directory is never mentioned, so it's NOT included.

# Set the PYTHONPATH for the code we copied
ENV PYTHONPATH="/code"

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]