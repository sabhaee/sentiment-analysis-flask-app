# To include the GPU drivers
# FROM nvidia/cuda:11.8-base
# CMD nvidia-smi

FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /api

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files to the container
COPY . .

# Expose the port for the Flask app (if applicable)
EXPOSE 5000

# Set the command to run the Flask app
CMD ["gunicorn","--config", "gunicorn_config.py", "wsgi:app" ]




