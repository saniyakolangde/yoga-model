# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     libgl1-mesa-glx  # This installs libGL.so.1
apt-get update && apt-get install libgl1


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
