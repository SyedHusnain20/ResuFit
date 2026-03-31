# Use official Python 3.10 slim image as base
# "slim" = smaller image size, faster builds
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /code

# Copy requirements first — Docker caches this layer
# If requirements.txt hasn't changed, pip install is skipped on rebuild
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Command to start the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]