# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=1.6.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Create and set the working directory
WORKDIR /app

# Copy only the dependency files first (to leverage Docker cache)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies (without creating virtualenv)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of the application code
COPY . /app

# Expose the port your app will run on
EXPOSE 8095

# Set the environment variable for the port
ENV PORT=8095

# Run the application using Taskipy
CMD ["poetry", "run", "task", "recall-svc-backend"]